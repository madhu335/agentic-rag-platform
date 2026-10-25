package com.example.airagassistant.domain.vehicle.service;

import com.example.airagassistant.rag.SearchHit;
import com.example.airagassistant.rag.VectorRecord;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowCallbackHandler;
import org.springframework.stereotype.Repository;

import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Types;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reads and writes the vehicle_attributes table (see schema.sql).
 */
@Slf4j
@Repository
@RequiredArgsConstructor
public class VehicleAttributeRepository {

    private final JdbcTemplate jdbcTemplate;

    private static final String UPSERT = """
            INSERT INTO vehicle_attributes (
                vehicle_id, fuel_type, drivetrain, body_style, is_electric,
                horsepower, torque, mpg_city, mpg_highway, mpg_combined, electric_range,
                msrp, annual_fuel_cost, maintenance_per_year, total_five_year_cost,
                resale_value_pct, nhtsa_overall, recall_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())
            ON CONFLICT (vehicle_id) DO UPDATE SET
                fuel_type = EXCLUDED.fuel_type,
                drivetrain = EXCLUDED.drivetrain,
                body_style = EXCLUDED.body_style,
                is_electric = EXCLUDED.is_electric,
                horsepower = EXCLUDED.horsepower,
                torque = EXCLUDED.torque,
                mpg_city = EXCLUDED.mpg_city,
                mpg_highway = EXCLUDED.mpg_highway,
                mpg_combined = EXCLUDED.mpg_combined,
                electric_range = EXCLUDED.electric_range,
                msrp = EXCLUDED.msrp,
                annual_fuel_cost = EXCLUDED.annual_fuel_cost,
                maintenance_per_year = EXCLUDED.maintenance_per_year,
                total_five_year_cost = EXCLUDED.total_five_year_cost,
                resale_value_pct = EXCLUDED.resale_value_pct,
                nhtsa_overall = EXCLUDED.nhtsa_overall,
                recall_count = EXCLUDED.recall_count,
                updated_at = now()
            """;

    public void upsertAll(List<VehicleAttributes> rows) {
        if (rows == null || rows.isEmpty()) return;
        jdbcTemplate.batchUpdate(UPSERT, rows, 500, VehicleAttributeRepository::bind);
    }

    private static void bind(PreparedStatement ps, VehicleAttributes a) throws SQLException {
        int i = 1;
        ps.setString(i++, a.vehicleId());
        ps.setString(i++, a.fuelType());
        ps.setString(i++, a.drivetrain());
        ps.setString(i++, a.bodyStyle());
        ps.setBoolean(i++, a.electric());
        setInt(ps, i++, a.horsepower());
        setInt(ps, i++, a.torque());
        setInt(ps, i++, a.mpgCity());
        setInt(ps, i++, a.mpgHighway());
        setInt(ps, i++, a.mpgCombined());
        setInt(ps, i++, a.electricRange());
        setInt(ps, i++, a.msrp());
        setInt(ps, i++, a.annualFuelCost());
        setInt(ps, i++, a.maintenancePerYear());
        setInt(ps, i++, a.totalFiveYearCost());
        if (a.resaleValuePct() == null) ps.setNull(i++, Types.DOUBLE); else ps.setDouble(i++, a.resaleValuePct());
        setInt(ps, i++, a.nhtsaOverall());
        ps.setInt(i, a.recallCount());
    }

    private static void setInt(PreparedStatement ps, int idx, Integer v) throws SQLException {
        if (v == null) ps.setNull(idx, Types.INTEGER); else ps.setInt(idx, v);
    }

    /**
     * Rebuilds attributes for every vehicle already in the database by parsing its chunks.
     * Safe to re-run: upserts by vehicle_id. Returns the number of vehicles processed.
     */
    public int backfillFromChunks() {
        Map<String, Map<Integer, String>> byVehicle = new LinkedHashMap<>();
        jdbcTemplate.query(
                "SELECT doc_id, chunk_index, content FROM document_chunks WHERE doc_type = 'vehicle' ORDER BY doc_id, chunk_index",
                (RowCallbackHandler) rs -> {
                    byVehicle.computeIfAbsent(rs.getString(1), k -> new HashMap<>())
                            .put(rs.getInt(2), rs.getString(3));
                });

        List<VehicleAttributes> rows = new ArrayList<>(byVehicle.size());
        byVehicle.forEach((id, chunks) -> rows.add(VehicleAttributeExtractor.extract(id, chunks)));
        upsertAll(rows);
        log.info("vehicle_attributes backfill: {} vehicles", rows.size());
        return rows.size();
    }

    /**
     * Runs a structured query and returns one chunk per matching vehicle, in rank order.
     * The chunk is the one that states the queried value (performance for hp, ownership
     * for costs, ...), falling back to the vehicle's lowest chunk if that one is missing.
     */
    public List<SearchHit> search(StructuredVehicleQuery q, int limit) {
        StructuredVehicleQuery.Field f = q.field();
        String col = f.column;   // enum constant, never user input

        StringBuilder where = new StringBuilder(col + " IS NOT NULL");
        List<Object> args = new ArrayList<>();
        if (f.excludesElectric()) where.append(" AND NOT is_electric");
        if (q.isThreshold()) {
            where.append(" AND ").append(col).append(' ').append(q.sqlOperator()).append(" ?");
            args.add(q.value());
        }
        String order = col + (q.descending() ? " DESC" : " ASC") + ", vehicle_id";

        String sql = """
                WITH ranked AS (
                    SELECT vehicle_id, row_number() OVER (ORDER BY %s) AS rnk
                    FROM vehicle_attributes
                    WHERE %s
                    ORDER BY %s
                    LIMIT ?
                )
                SELECT DISTINCT ON (r.rnk) r.rnk, c.doc_id, c.chunk_index, c.content
                FROM ranked r
                JOIN document_chunks c ON c.doc_id = r.vehicle_id AND c.doc_type = 'vehicle'
                ORDER BY r.rnk, (c.chunk_index = ?) DESC, c.chunk_index
                """.formatted(order, where, order);
        args.add(limit);
        args.add(f.chunkIndex);

        List<SearchHit> hits = jdbcTemplate.query(sql, (rs, n) -> {
            String docId = rs.getString("doc_id");
            int chunkIndex = rs.getInt("chunk_index");
            int rank = rs.getInt("rnk");
            VectorRecord record = new VectorRecord(
                    Math.abs((long) docId.hashCode()), chunkIndex,
                    docId + ":" + chunkIndex, rs.getString("content"), List.of());
            // Rank-derived score in (0, 1]: callers only rely on order; this keeps scores monotonic.
            return new SearchHit(record, 1.0 / rank);
        }, args.toArray());

        log.info("Structured fleet query field={} op={} value={} desc={} -> {} vehicles",
                f, q.op(), q.value(), q.descending(), hits.size());
        return hits;
    }
}
