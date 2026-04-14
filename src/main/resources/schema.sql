create table if not exists agent_sessions (
                                              session_id varchar(100) primary key,
                                              original_user_request text,
                                              current_user_request text,
                                              doc_id varchar(255),
                                              state_json jsonb not null,
                                              created_at timestamptz not null,
                                              updated_at timestamptz not null,
                                              version bigint not null default 0
);

create index if not exists idx_agent_sessions_updated_at
    on agent_sessions(updated_at desc);

create index if not exists idx_agent_sessions_doc_id
    on agent_sessions(doc_id);