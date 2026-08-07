# API Documentation

## `RailwaySurveillanceSystem`
Central manager for loading models, vector databases, and restricted zone polygons.

### Methods
- `add_criminal_to_db(name, image_path)`: Register criminal face embedding.
- `add_worker_to_db(name, image_path)`: Register worker face embedding.
- `add_zone(zone_name, polygon_points, zone_type="restricted")`: Configure security polygon.

## `UnifiedPipeline`
High-speed processing engine linking all surveillance modules.

### Methods
- `process_frame(frame, ...)`: Run active modules and return annotated frame with analytics.
