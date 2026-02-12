# Oracle Mock Table Schema

This directory stores CSV files that mimic Oracle source tables for a predictive parts recommendation demo.

## Table: `dealer_cases`
- Grain: one dealer support case per row.
- Columns:
  - `case_id` (PK): case identifier.
  - `dealer_id`: dealer identifier.
  - `asset_id` (FK -> `assets.asset_id`): affected machine.
  - `opened_date`: case open date (ISO date).
  - `symptom_text`: free-text dealer complaint.
  - `region`: dealer region.
  - `severity`: integer from 1 (low) to 5 (high).

## Table: `assets`
- Grain: one machine/asset per row.
- Columns:
  - `asset_id` (PK): asset identifier.
  - `model_family`: model family name.
  - `equipment_type`: machine class (tractor-like in this demo).
  - `age_days`: age in days.
  - `usage_hours_total`: total usage hours.

## Table: `parts`
- Grain: one part type per row.
- Columns:
  - `part_id` (PK): part identifier.
  - `part_name`: display name.
  - `unit_cost`: replacement cost per unit.
  - `replacement_time_hours`: service time per replacement.
  - `expedite_ship_cost`: expedited shipment cost per unit.
  - `satisfaction_penalty_per_day`: customer-impact penalty per day of downtime.

## Table: `tasks`
- Grain: one task type per row.
- Columns:
  - `task_type` (PK): operation type.
  - `intensity_scalar`: baseline task intensity.

## Table: `task_part_strain`
- Grain: task/part relationship.
- Columns:
  - `task_type` (FK -> `tasks.task_type`)
  - `part_id` (FK -> `parts.part_id`)
  - `strain_multiplier`: how much task increases part wear/failure tendency.

## Table: `events`
- Grain: one operating event tied to a case.
- Columns:
  - `event_id` (PK): event identifier.
  - `case_id` (FK -> `dealer_cases.case_id`): source case.
  - `task_type` (FK -> `tasks.task_type`): work type performed.
  - `duration_days`: event duration.
  - `environment_score`: stress factor from environment (0-1).
  - `avg_load_factor`: average load factor.
  - `is_demo`: whether event belongs to demo slice.

## Table: `failures`
- Grain: part failures by day within an event.
- Columns:
  - `failure_id` (PK): failure row identifier.
  - `event_id` (FK -> `events.event_id`)
  - `part_id` (FK -> `parts.part_id`)
  - `day_of_event`: event day when failure occurred.
  - `failed_qty`: quantity failed on that day.

## Table: `parts_used`
- Grain: total quantity used per event and part.
- Columns:
  - `event_id` (FK -> `events.event_id`)
  - `part_id` (FK -> `parts.part_id`)
  - `used_qty`: quantity consumed during service for event.

## Table: `trip_outcomes`
- Grain: service trip outcome metrics per event.
- Columns:
  - `event_id` (PK/FK -> `events.event_id`)
  - `had_stockout`: whether stockout happened.
  - `extra_trip_required`: whether follow-up trip was needed.
  - `downtime_hours`: downtime hours observed.
  - `total_cost`: total event cost.

## Table: `inventory_snapshot`
- Grain: region + part inventory snapshot.
- Columns:
  - `region`: region name.
  - `part_id` (FK -> `parts.part_id`)
  - `on_hand_qty`: quantity in stock.
  - `last_updated`: snapshot date.

## Generation Notes
- `symptom_text` is correlated with likely failing parts with controlled noise.
- Part failures are sampled from:
  - `N_i ~ Poisson(lambda_i * duration_days)`
  - `lambda_i = base_lambda(part) * f(age_days) * g(task_strain) * h(load_factor, environment_score) * noise(part,event)`
- Data is reproducible using script seed.
