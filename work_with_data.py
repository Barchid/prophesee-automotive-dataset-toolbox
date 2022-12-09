from src.io.box_loading import reformat_boxes
from src.io.box_filtering import filter_boxes
import numpy as np
from src.io.psee_loader import PSEELoader

TARGET_DIR = "data/gen1_formatted"


def main():
    # Find data

    # FOR EACH DATA
    for vid_path in range(len(10)):  # TODO: change
        # Load video and related bboxes
        video = PSEELoader(vid_path)
        ev = video.load_n_events(video.event_count())
        total_events = np.empty(
            len(ev),
            dtype=np.dtype(
                [("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)]
            ),
        )
        total_events["x"] = ev["x"]
        total_events["y"] = ev["y"]
        total_events["t"] = ev["t"]
        total_events["p"] = ev["p"]
        del video

        total_bbox = np.load(vid_path.replace("_td.dat", "_bbox.npy"))
        total_bbox = reformat_boxes(total_bbox)
        total_bbox = filter_boxes(
            total_bbox, skip_ts=0, min_box_diag=30, min_box_side=10
        )

        ts_bbox = np.unique(total_bbox["t"])
        ts_bbox = np.sort(ts_bbox)  # sort timestamps
        start_ts = 0
        for i in range(1, len(ts_bbox)):
            end_ts = ts_bbox[i]

            # events save
            mask = (total_events["t"] >= start_ts) & (total_events["t"] <= end_ts)
            events = total_events[mask]
            events["t"] = events["t"] - events["t"].min()  # begin ts at 0
            np.save(
                f"{TARGET_DIR}/{vid_path.replace('_td.dat', '')}_{str(end_ts).zfill(3)}.npy"
            )

            # bbox save
            mask = (total_bbox["t"] >= start_ts) & (total_bbox["t"] <= end_ts)
            bbox = total_bbox[mask]
            np.save(
                f"{TARGET_DIR}/{vid_path.replace('_td.dat', '')}_{str(end_ts).zfill(3)}_bbox.npy"
            )

            start_ts = end_ts


if __name__ == "__main__":
    main()
