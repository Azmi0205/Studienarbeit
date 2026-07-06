
import os
import time
import csv
import numpy as np
import cv2

os.chdir(r"C:\Users\azmib\Desktop\Python")
from BosonSDK import *

COM_PORT = 'COM3'          # your camera's COM port
VIDEO_INDEX = 1            # your camera's video device index
OUTPUT_ROOT = 'captures'
FRAMES_PER_BATCH = 50

# Match these to your Boson configuration + CCI telemetry setting.
# 640 config, no telemetry rows: 640x512
# 640 config, with telemetry:    640x514
# 320 config, no telemetry rows: 320x256
# 320 config, with telemetry:    320x258
FRAME_WIDTH = 320
FRAME_HEIGHT = 256


def configure_camera(cam):
    cam.gaoSetAveragerState(FLR_ENABLE_E.FLR_DISABLE)           # FLR_DISABLE = 0, no frame averager
    cam.roicSetFrameSkip(0)             # no frame skip
    cam.radiometrySetTempStableEnable(FLR_ENABLE_E.FLR_ENABLE)  # FLR_ENABLE = 1, T-stable on
    cam.TLinearSetControl(FLR_ENABLE_E.FLR_ENABLE)              # T-Linear on
    print("Camera configured: no averager, no frame skip, T-stable ON, T-Linear ON")
    print("Reminder: the Pre-AGC (Y16) UVC output format is requested by the capture "
          "software (see open_video), not set via CCI.")


def get_fpa_temp_c(cam):
    result, temp_x10 = cam.bosonlookupFPATempDegCx10()
    return temp_x10 / 10.0


def open_video(width=FRAME_WIDTH, height=FRAME_HEIGHT):
    cap = cv2.VideoCapture(VIDEO_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video device index {VIDEO_INDEX}")

    # Explicitly request the Pre-AGC 16-bit UVC format (4CC "Y16 ")
    fourcc = cv2.VideoWriter_fourcc('Y', '1', '6', ' ')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # do not auto-convert to BGR, keep raw

    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Negotiated FourCC: '{actual_fourcc_str}', size: {actual_w}x{actual_h}")

    if actual_fourcc_str.strip() != "Y16":
        print("WARNING: Camera did not negotiate Y16 format. "
              "You may be receiving AGC-processed/YUV data instead of Pre-AGC.")

    return cap


def read_raw16_frame(cap, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    ret, frame = cap.read()
    if not ret:
        return False, None

    if frame.dtype == np.uint16:
        raw16 = frame
        if raw16.ndim == 3:
            raw16 = raw16.reshape(height, width)
    else:
        # frame arrives as uint8, packed 2 bytes/pixel -> view as uint16, then reshape
        raw16 = frame.view(np.uint16).reshape(height, width)

    return True, raw16


def capture_batch(cam, batch_name, run_ffc=False):
    if run_ffc:
        print("Running FFC before this batch...")
        cam.bosonRunFFC()
        time.sleep(1)

    folder = os.path.join(OUTPUT_ROOT, batch_name)
    os.makedirs(folder, exist_ok=True)

    temp_c = get_fpa_temp_c(cam)
    log_path = os.path.join(OUTPUT_ROOT, "temperature_log.csv")
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["batch_name", "timestamp", "fpa_temp_c", "num_frames"])
        writer.writerow([batch_name, time.strftime("%Y-%m-%d %H:%M:%S"), temp_c, FRAMES_PER_BATCH])

    cap = open_video()

    count = 0
    while count < FRAMES_PER_BATCH:
        ret, raw16 = read_raw16_frame(cap)
        if not ret:
            print("Frame grab failed, retrying...")
            continue

        filename = os.path.join(folder, f"frame_{count:03d}.npy")
        np.save(filename, raw16)

        if count == 0:
            print(f"  First frame stats -> min: {raw16.min()}, "
                  f"max: {raw16.max()}, mean: {raw16.mean():.1f}")

        count += 1

    cap.release()
    print(f"Batch '{batch_name}' done: {count} frames saved to {folder}, FPA temp = {temp_c:.1f} C")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    cam = CamAPI.pyClient(manualport=COM_PORT)
    configure_camera(cam)

    print("\nCommands:")
    print("  n  -> capture 50 pictures, no FFC")
    print("  f  -> run FFC, then capture 50 pictures")
    print("  q  -> quit")

    batch_counter = 0
    try:
        while True:
            cmd = input("\nEnter command (n/f/q): ").strip().lower()
            if cmd == 'q':
                break
            elif cmd == 'n':
                batch_counter += 1
                capture_batch(cam, f"batch_{batch_counter:02d}_noffc", run_ffc=False)
            elif cmd == 'f':
                batch_counter += 1
                capture_batch(cam, f"batch_{batch_counter:02d}_ffc", run_ffc=True)
            else:
                print("Unknown command. Use n, f, or q.")
    finally:
        cam.Close()
        print("Camera connection closed.")


if __name__ == "__main__":
    main()
