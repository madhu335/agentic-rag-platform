package com.example.airagassistant.domain.vehicle.service;

import com.example.airagassistant.domain.vehicle.service.StructuredVehicleQuery.Field;
import com.example.airagassistant.domain.vehicle.service.StructuredVehicleQuery.Op;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

class StructuredVehicleQueryTest {

    @ParameterizedTest(name = "{0}")
    @CsvSource(delimiter = '|', value = {
            // query                                         | field                | op          | value  | desc
            "which vehicles have over 500 horsepower          | HORSEPOWER           | GT          | 500    | true",
            "vehicles under $40000                            | MSRP                 | LT          | 40000  | false",
            "vehicles priced below $35,000                    | MSRP                 | LT          | 35000  | false",
            "cars under 40k                                   | MSRP                 | LT          | 40000  | false",
            "vehicles with at least 40 mpg fuel economy       | MPG                  | GTE         | 40     | true",
            "vehicles with 300 hp or more                     | HORSEPOWER           | GTE         | 300    | true",
            "which car has the most torque                    | TORQUE               | SUPERLATIVE |        | true",
            "which vehicle has the best resale value          | RESALE_VALUE_PCT     | SUPERLATIVE |        | true",
            "which vehicles are most expensive to maintain    | MAINTENANCE_PER_YEAR | SUPERLATIVE |        | true",
            "least expensive to maintain                      | MAINTENANCE_PER_YEAR | SUPERLATIVE |        | false",
            "vehicle with lowest total 5 year ownership cost  | TOTAL_FIVE_YEAR_COST | SUPERLATIVE |        | false",
            "cheapest vehicle                                 | MSRP                 | SUPERLATIVE |        | false",
            "most affordable SUV                              | MSRP                 | SUPERLATIVE |        | false",
            "most fuel efficient car                          | MPG                  | SUPERLATIVE |        | true",
            "longest range EV                                 | ELECTRIC_RANGE       | SUPERLATIVE |        | true",
    })
    void parsesStructuredQuestions(String query, Field field, Op op, Double value, boolean desc) {
        StructuredVehicleQuery q = StructuredVehicleQuery.parse(query).orElseThrow();
        assertEquals(field, q.field());
        assertEquals(op, q.op());
        assertEquals(value, q.value());
        assertEquals(desc, q.descending());
    }

    @ParameterizedTest(name = "{0}")
    @ValueSource(strings = {
            // single-vehicle questions and subjective ones must stay on hybrid text search
            "how much does it cost to own the BMW M3 over 5 years",
            "what does the 30000 mile service cost on the BMW M3",
            "BMW M3 CS trim price",
            "what is the price of the M3",
            "bmw-m3-2025-competition which has the most horsepower",
            "good family car with low running costs",
            "fast sports car that wont break the bank",
            "electric vehicles",
            "AWD vehicles",
            "best car for someone who loves track days but needs daily practicality",
            "how much will I spend on gas per year",
            "xyzabc vehicle",
            "M3",
    })
    void leavesOtherQuestionsToHybridSearch(String query) {
        assertTrue(StructuredVehicleQuery.parse(query).isEmpty(), () -> "should not be structured: " + query);
    }
}
