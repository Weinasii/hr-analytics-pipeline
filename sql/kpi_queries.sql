-- ════════════════════════════════════════════════════════════════════════════
-- hr_analytics · SQL KPI Library
-- Compatible with PostgreSQL 14+ · Adapt schema prefix as needed
-- ════════════════════════════════════════════════════════════════════════════

-- ─────────────────────────────────────────────────────────────────────────────
-- SCHEMA
-- ─────────────────────────────────────────────────────────────────────────────

CREATE SCHEMA IF NOT EXISTS hr;

CREATE TABLE IF NOT EXISTS hr.employees (
    employee_id             VARCHAR(20)     PRIMARY KEY,
    full_name               VARCHAR(100),
    age                     SMALLINT        CHECK (age BETWEEN 16 AND 80),
    gender                  VARCHAR(20),
    department              VARCHAR(50)     NOT NULL,
    job_level               VARCHAR(30),
    education_level         VARCHAR(30),
    hire_date               DATE            NOT NULL,
    departure_date          DATE,
    salary_eur              NUMERIC(10,2)   CHECK (salary_eur > 0),
    performance_score       SMALLINT        CHECK (performance_score BETWEEN 1 AND 5),
    satisfaction_score      NUMERIC(3,1)    CHECK (satisfaction_score BETWEEN 1.0 AND 5.0),
    overtime_hours_monthly  SMALLINT        DEFAULT 0,
    distance_km             SMALLINT,
    training_hours_ytd      SMALLINT        DEFAULT 0,
    months_since_promotion  SMALLINT        DEFAULT 0,
    manager_rating          SMALLINT        CHECK (manager_rating BETWEEN 1 AND 5),
    absences_ytd            SMALLINT        DEFAULT 0,
    attrition               BOOLEAN         DEFAULT FALSE,
    created_at              TIMESTAMPTZ     DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_employees_department ON hr.employees (department);
CREATE INDEX IF NOT EXISTS idx_employees_hire_date  ON hr.employees (hire_date);
CREATE INDEX IF NOT EXISTS idx_employees_attrition  ON hr.employees (attrition);


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. OVERALL ATTRITION RATE
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    COUNT(*)                                                        AS total_employees,
    SUM(attrition::INT)                                             AS leavers,
    ROUND(AVG(attrition::NUMERIC) * 100, 2)                        AS attrition_rate_pct
FROM hr.employees;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. MONTHLY ATTRITION TREND (last 12 months)
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    DATE_TRUNC('month', departure_date)                            AS month,
    COUNT(*)                                                        AS departures,
    ROUND(
        COUNT(*) * 100.0 /
        NULLIF(SUM(COUNT(*)) OVER (ORDER BY DATE_TRUNC('month', departure_date) ROWS BETWEEN 11 PRECEDING AND CURRENT ROW), 0),
        2
    )                                                               AS rolling_12m_rate_pct
FROM hr.employees
WHERE
    attrition = TRUE
    AND departure_date >= DATE_TRUNC('month', NOW()) - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', departure_date)
ORDER BY month;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. ATTRITION BY DEPARTMENT — with benchmark comparison
-- ─────────────────────────────────────────────────────────────────────────────

WITH dept_stats AS (
    SELECT
        department,
        COUNT(*)                                    AS headcount,
        SUM(attrition::INT)                         AS leavers,
        ROUND(AVG(attrition::NUMERIC) * 100, 2)    AS attrition_rate_pct
    FROM hr.employees
    GROUP BY department
),
global_rate AS (
    SELECT ROUND(AVG(attrition::NUMERIC) * 100, 2) AS global_avg
    FROM hr.employees
)
SELECT
    d.department,
    d.headcount,
    d.leavers,
    d.attrition_rate_pct,
    g.global_avg                                    AS company_avg_pct,
    ROUND(d.attrition_rate_pct - g.global_avg, 2)  AS delta_vs_avg,
    CASE
        WHEN d.attrition_rate_pct > g.global_avg + 5 THEN '🔴 High'
        WHEN d.attrition_rate_pct < g.global_avg - 5 THEN '🟢 Low'
        ELSE '🟡 Normal'
    END                                             AS risk_flag
FROM dept_stats d, global_rate g
ORDER BY d.attrition_rate_pct DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. GENDER PAY GAP (unadjusted)
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    department,
    gender,
    COUNT(*)                                AS headcount,
    ROUND(AVG(salary_eur), 0)              AS avg_salary_eur,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary_eur), 0) AS median_salary_eur
FROM hr.employees
WHERE gender IN ('M', 'F')
GROUP BY department, gender
ORDER BY department, gender;

-- Summary: company-wide gender pay gap %
SELECT
    ROUND(
        (AVG(CASE WHEN gender = 'M' THEN salary_eur END)
         - AVG(CASE WHEN gender = 'F' THEN salary_eur END))
        / NULLIF(AVG(CASE WHEN gender = 'M' THEN salary_eur END), 0) * 100,
        2
    ) AS gender_pay_gap_pct
FROM hr.employees
WHERE gender IN ('M', 'F');


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. ABSENTEEISM RATE
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    department,
    ROUND(AVG(absences_ytd), 1)                        AS avg_absences_days,
    ROUND(AVG(absences_ytd) / 220.0 * 100, 2)         AS absenteeism_rate_pct,
    COUNT(*) FILTER (WHERE absences_ytd > 15)          AS high_absentees,
    COUNT(*)                                            AS headcount
FROM hr.employees
GROUP BY department
ORDER BY absenteeism_rate_pct DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 6. HIGH-RISK EMPLOYEE IDENTIFICATION (rule-based)
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    employee_id,
    department,
    job_level,
    ROUND(tenure_months / 12.0, 1)             AS tenure_years,
    satisfaction_score,
    overtime_hours_monthly,
    months_since_promotion,
    manager_rating,
    -- Composite risk score (0–100)
    ROUND(
        (LEAST(overtime_hours_monthly, 40) / 40.0 * 30)
        + ((5.0 - satisfaction_score) / 4.0 * 30)
        + (LEAST(months_since_promotion, 36) / 36.0 * 20)
        + ((180 - LEAST(tenure_months, 180)) / 180.0 * 20),
        1
    )                                           AS risk_score
FROM hr.employees
WHERE attrition = FALSE  -- only active employees
ORDER BY risk_score DESC
LIMIT 50;


-- ─────────────────────────────────────────────────────────────────────────────
-- 7. COMPENSATION EQUITY — P10/P50/P90 by job level
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    job_level,
    COUNT(*)                                                                    AS headcount,
    ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY salary_eur), 0)        AS p10_salary,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY salary_eur), 0)        AS p50_salary,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY salary_eur), 0)        AS p90_salary,
    ROUND(
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY salary_eur) /
        NULLIF(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY salary_eur), 0),
        2
    )                                                                           AS p90_p10_ratio
FROM hr.employees
GROUP BY job_level
ORDER BY p50_salary DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 8. TRAINING EFFECTIVENESS — attrition vs training hours
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    CASE
        WHEN training_hours_ytd = 0      THEN '0h (none)'
        WHEN training_hours_ytd <= 10    THEN '1–10h'
        WHEN training_hours_ytd <= 30    THEN '11–30h'
        ELSE '30h+'
    END                                                 AS training_bucket,
    COUNT(*)                                            AS headcount,
    ROUND(AVG(attrition::NUMERIC) * 100, 2)           AS attrition_rate_pct,
    ROUND(AVG(satisfaction_score), 2)                  AS avg_satisfaction
FROM hr.employees
GROUP BY training_bucket
ORDER BY attrition_rate_pct DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 9. HEADCOUNT GROWTH — cumulative monthly
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    DATE_TRUNC('month', hire_date)  AS month,
    COUNT(*)                        AS new_hires,
    SUM(COUNT(*)) OVER (ORDER BY DATE_TRUNC('month', hire_date))  AS cumulative_headcount
FROM hr.employees
GROUP BY DATE_TRUNC('month', hire_date)
ORDER BY month;


-- ─────────────────────────────────────────────────────────────────────────────
-- 10. MANAGER EFFECTIVENESS INDEX
-- ─────────────────────────────────────────────────────────────────────────────

-- Employees grouped by their manager_rating (proxy for manager quality)
SELECT
    manager_rating,
    COUNT(*)                                    AS team_size,
    ROUND(AVG(attrition::NUMERIC) * 100, 2)   AS attrition_rate_pct,
    ROUND(AVG(satisfaction_score), 2)          AS avg_satisfaction,
    ROUND(AVG(performance_score), 2)           AS avg_performance,
    ROUND(AVG(absences_ytd), 1)               AS avg_absences
FROM hr.employees
GROUP BY manager_rating
ORDER BY manager_rating;
