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
-- Typed per-vehicle facts for structured fleet queries ("over 500 hp", "most torque").
-- Populated at ingestion and by POST /vehicles/admin/attributes/backfill.
-- Values are extracted from the indexed chunk text; NULL means "not known" and never matches a filter.
create table if not exists vehicle_attributes (
    vehicle_id            text primary key,
    fuel_type             text,
    drivetrain            text,
    body_style            text,
    is_electric           boolean not null default false,
    horsepower            int,
    torque                int,
    mpg_city              int,
    mpg_highway           int,
    mpg_combined          int,
    electric_range        int,
    msrp                  int,
    annual_fuel_cost      int,
    maintenance_per_year  int,
    total_five_year_cost  int,
    resale_value_pct      double precision,
    nhtsa_overall         int,
    recall_count          int not null default 0,
    updated_at            timestamptz not null default now()
);
