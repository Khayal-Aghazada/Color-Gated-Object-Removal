# python -m apps.object_remove
# python -m apps.object_remove --src assets/demo_video/cars.mp4 --auto_frames 30



"""
Object removal with fixed camera + clean plate, with optional color gating and view modes.

Flow
----
- Build or load a clean plate.
- Build mask:
    * color: HSV mask only (e.g., purple cloth).
    * diff:  absdiff(frame, plate).
    * both:  color AND diff.
- Fill masked pixels by copying from plate (default) or inpainting.

Keys
----
[p] capture plate from current frame
[s] save current plate to --save_plate
[q]/[ESC] quit
"""
import argparse
import cv2
import numpy as np
from modules import Camera
from modules.bg import CleanPlate, ForegroundExtractor, PlateFiller


# Default HSV for purple (adjust for your lighting).
PURPLE_DEFAULT_LOW  = (120,  90,  60)
PURPLE_DEFAULT_HIGH = (155, 255, 255)


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=0, help="0/1 webcam or path")
    p.add_argument("--wait", type=int, default=1)
    p.add_argument("--mirror", type=int, default=1)

    # Plate options
    p.add_argument("--plate", default="", help="optional path to clean-plate image")
    p.add_argument("--auto_frames", type=int, default=0,
                   help="if >0, take median of first N frames as plate")
    p.add_argument("--save_plate", default="data/raw/clean_plate.jpg")

    # Foreground differencing
    p.add_argument("--thresh", type=int, default=30)
    p.add_argument("--k_open", type=int, default=3)
    p.add_argument("--k_close", type=int, default=7)

    # Mask building
    p.add_argument("--mask_mode", choices=["color", "diff", "both"], default="color",
                   help="how to build the removal mask")
    p.add_argument("--hsv", nargs=6, type=int, metavar=("H1","S1","V1","H2","S2","V2"),
                   help="override HSV low/high for color gating")

    # Fill method
    p.add_argument("--method", choices=["copy","inpaint"], default="copy")

    # View mode
    p.add_argument("--view", choices=["out","triptych","mask","frame"], default="out",
                   help="what to display")
    return p.parse_args()


def hsv_mask(frame: np.ndarray, low: tuple, high: tuple) -> np.ndarray:
    """Return uint8 mask (0/255) for pixels within HSV range."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
    
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def save_image(path: str, img: np.ndarray) -> None:
    """Safe write to disk."""
    cv2.imwrite(path, img)b


def main():
    """Run the object removal demo."""
    args = parse_args()
    cam = Camera(args.src, mirror=bool(args.mirror), wait=args.wait)

    cp = CleanPlate()
    if args.plate:
        img = cv2.imread(args.plate, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(args.plate)
        cp.set(img)
        print(f"Loaded plate from: {args.plate}")
    if not cp.available() and args.auto_frames > 0:
        frames = []
        for i, frame in enumerate(cam.iterate()):
            frames.append(frame.copy())
            cam.show("Object Remove", frame)
            if i + 1 >= args.auto_frames:
                break
        plate = cp.from_median(frames)
        save_image(args.save_plate, plate)
        print(f"Auto-built plate from {len(frames)} frames → {args.save_plate}")

    fe = ForegroundExtractor(thresh=args.thresh, k_open=args.k_open, k_close=args.k_close)
    filler = PlateFiller(method=args.method, inpaint_radius=3)

    # HSV range
    if args.hsv is None:
        low, high = PURPLE_DEFAULT_LOW, PURPLE_DEFAULT_HIGH
    else:
        low = (args.hsv[0], args.hsv[1], args.hsv[2])
        high = (args.hsv[3], args.hsv[4], args.hsv[5])

    for frame in cam.iterate():
        if not cp.available():
            cv2.putText(frame, "Press 'p' to capture clean plate", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "Press 'p' to capture clean plate", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            view = frame
            cam.show("Object Remove", view)
        else:
            # Build masks
            diff_mask = fe.mask(frame, cp.plate) if args.mask_mode in ("diff", "both") else None
            color_mask = hsv_mask(frame, low, high) if args.mask_mode in ("color", "both") else None

            if args.mask_mode == "color":
                final_mask = color_mask
            elif args.mask_mode == "diff":
                final_mask = diff_mask
            else:
                if color_mask is None or diff_mask is None:
                    final_mask = np.zeros(frame.shape[:2], np.uint8)
                else:
                    final_mask = cv2.bitwise_and(diff_mask, color_mask)

            if final_mask is None:
                final_mask = np.zeros(frame.shape[:2], np.uint8)

            # strengthen edges slightly
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            final_mask = cv2.medianBlur(final_mask, 5)

            out = filler.fill(frame, final_mask, cp.plate)

            # visualization
            if args.view == "triptych":
                m3 = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
                view = np.hstack([frame, m3, out])
            elif args.view == "mask":
                view = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
            elif args.view == "frame":
                view = frame
            else:  # "out"
                view = out
            cam.show("Object Remove", view)

        k = cam.wait_key()
        if k in (27, ord('q')):
            break
        if k == ord('p'):
            cp.set(frame)
            save_image(args.save_plate, cp.plate)
            print(f"Captured plate → {args.save_plate}")
        if k == ord('s') and cp.available():
            save_image(args.save_plate, cp.plate)
            print(f"Saved plate → {args.save_plate}")

    cam.release()


if __name__ == "__main__":
    main()
