package com.example.airagassistant.domain.vehicle.service;

import com.example.airagassistant.rag.SearchHit;
import com.example.airagassistant.rag.VectorRecord;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class VariantCollapserTest {

    private static SearchHit hit(String chunkId) {
        return new SearchHit(new VectorRecord(0, 0, chunkId, "", List.of()), 1.0);
    }

    private static List<String> ids(List<SearchHit> hits) {
        return hits.stream().map(h -> h.record().id()).toList();
    }

    @Test
    void familyIsMakeAndModelWithoutYear() {
        assertEquals("toyota-corolla", VariantCollapser.familyOf("toyota-corolla-2024-hatchback-fwd-2-0l-4-cylinder-epa47345"));
        assertEquals("toyota-corolla", VariantCollapser.familyOf("toyota-corolla-2025-hatchback-fwd-2-0l-4-cylinder-epa47996"));
        assertEquals("rivian-r1t", VariantCollapser.familyOf("rivian-r1t-2025-adventure"));
        assertEquals("ford-f-150-lightning", VariantCollapser.familyOf("ford-f-150-lightning-2024-4wd-extended-range-epa47818"));
        assertEquals("bmw-m3", VariantCollapser.familyOf("bmw-m3-2025-competition"));
        // the year in "model 3" style names is not a 4-digit year, and ids without a year stand alone
        assertEquals("tesla-model3", VariantCollapser.familyOf("tesla-model3-2025-long-range"));
        assertEquals("baseline-test-vehicle", VariantCollapser.familyOf("baseline-test-vehicle"));
    }

    @Test
    void keepsBestRankedVariantPerFamily_andAllItsChunks() {
        List<SearchHit> in = List.of(
                hit("rivian-r1t-2025-all-terrain-epa1:2"),
                hit("rivian-r1t-2025-all-terrain-epa1:6"),   // same vehicle: kept
                hit("rivian-r1t-2024-dual-max-epa2:2"),       // same family, other variant: dropped
                hit("ford-f-150-lightning-2024-4wd-epa3:2"),
                hit("rivian-r1t-2025-tri-max-epa4:2"),        // dropped
                hit("gmc-sierra-2025-ev-4wd-epa5:2"));

        List<SearchHit> out = VariantCollapser.collapse(in, 4);

        assertEquals(List.of(
                "rivian-r1t-2025-all-terrain-epa1:2",
                "rivian-r1t-2025-all-terrain-epa1:6",
                "ford-f-150-lightning-2024-4wd-epa3:2",
                "gmc-sierra-2025-ev-4wd-epa5:2"), ids(out));
    }

    @Test
    void refillsFromDroppedVariantsWhenTooFewFamilies() {
        List<SearchHit> in = List.of(
                hit("toyota-corolla-2024-hatchback-epa1:2"),
                hit("toyota-corolla-2025-hatchback-epa2:2"),
                hit("toyota-corolla-2024-sedan-epa3:2"));

        List<SearchHit> out = VariantCollapser.collapse(in, 2);

        assertEquals(List.of(
                "toyota-corolla-2024-hatchback-epa1:2",
                "toyota-corolla-2025-hatchback-epa2:2"), ids(out),
                "only one family: refill in rank order up to minResults");
    }

    @Test
    void emptyInputIsReturnedAsIs() {
        assertTrue(VariantCollapser.collapse(List.of(), 5).isEmpty());
    }
}
