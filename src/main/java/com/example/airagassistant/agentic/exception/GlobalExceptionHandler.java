package com.example.airagassistant.agentic.exception;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.time.Instant;
import java.util.Map;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(WorkflowStateException.class)
    public ResponseEntity<Map<String, Object>> handleWorkflowStateException(WorkflowStateException ex) {
        return ResponseEntity.unprocessableEntity().body(Map.of(
                "timestamp", Instant.now().toString(),
                "error", ex.getErrorCode(),
                "message", ex.getMessage()
        ));
    }

    // Validation errors from VehicleIngestionService, tool executors, etc.
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, Object>> handleIllegalArgument(IllegalArgumentException ex) {
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(Map.of(
                "timestamp", Instant.now().toString(),
                "error", "INVALID_REQUEST",
                "message", ex.getMessage()
        ));
    }

    // Unexpected runtime errors — log-worthy but don't leak stack traces
    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<Map<String, Object>> handleIllegalState(IllegalStateException ex) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(Map.of(
                "timestamp", Instant.now().toString(),
                "error", "INTERNAL_ERROR",
                "message", ex.getMessage()
        ));
    }
}
