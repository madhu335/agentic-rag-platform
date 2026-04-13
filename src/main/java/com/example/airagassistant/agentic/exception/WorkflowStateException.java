package com.example.airagassistant.agentic.exception;

public class WorkflowStateException extends RuntimeException {

    private final String errorCode;

    public WorkflowStateException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public String getErrorCode() {
        return errorCode;
    }
}