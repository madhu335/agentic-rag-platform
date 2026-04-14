package com.example.airagassistant.agentic.state;

import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface AgentSessionRepository extends JpaRepository<AgentSessionEntity, String> {
    List<AgentSessionEntity> findAllByOrderByUpdatedAtDesc(Pageable pageable);
}