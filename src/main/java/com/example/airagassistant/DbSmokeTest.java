package com.example.airagassistant;

import org.springframework.boot.CommandLineRunner;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class DbSmokeTest implements CommandLineRunner {
    private final JdbcTemplate jdbc;

    public DbSmokeTest(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    @Override
    public void run(String... args) {
        List<String> tables = jdbc.queryForList(
                "select table_name from information_schema.tables where table_schema='PUBLIC'",
                String.class
        );
        System.out.println("H2 tables: " + tables);
    }
}
