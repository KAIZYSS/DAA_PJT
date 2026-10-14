from backend.data_pipeline.storage.database import (
    init_pipeline_tables,
    save_data_source,
    save_validation_log,
    save_anomalies_batch,
    save_road_segments,
    get_all_road_segments,
    get_road_segment_by_id,
    get_all_data_sources,
    get_anomalies_filtered,
    update_segment_live_traffic
)
