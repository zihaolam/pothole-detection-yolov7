import base64, os, cv2, time
import json
from predict import build_model, detect, format_yolov5, wrap_detection
from uuid import uuid4


def show_coordinates(frame, index):
    coordinate_index = index % n_coordinates
    coordinate = coordinates[coordinate_index]
    cv2.putText(
        frame,
        f"[{round(coordinate[0], 4)}, {round(coordinate[1], 4)}]",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )


if __name__ == "_main_":
    video_name = "mixkit-potholes-in-a-rural-road-25208-large"
    cleaned_video_name = video_name.lower().replace(" ", "")
    dir_name = f"pothole_images/{cleaned_video_name}"
    os.makedirs(dir_name, exist_ok=True)
    with open("./coordinates.json") as f:
        coordinates = json.load(f)

    n_coordinates = len(coordinates)
    capture = cv2.VideoCapture(f"input/{video_name}.mp4")
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
    class_list = ["pothole"]

    is_cuda = False
    net = build_model(is_cuda)
    start = time.time_ns()
    frame_count = 0
    total_frames = 0
    fps = -1
    replaying = False
    frame_wait_count = 0
    image_index = 1
    index = 0

    while True:
        ret, frame = capture.read()

        if not ret:
            print("no video")
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            replaying = True
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputImage = format_yolov5(frame)
        outs = detect(inputImage, net)

        class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

        frame_count += 1
        total_frames += 1

        if frame_wait_count > 0:
            frame_wait_count -= 1

        for classid, confidence, box in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(
                frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1
            )
            cv2.putText(
                frame,
                class_list[classid],
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
            )

        if len(boxes) and not replaying and frame_wait_count <= 0:
            # cv2.imwrite(f"{dir_name}/{image_index}_##{len(boxes)}.jpg", frame)
            image_index += 1
            frame_wait_count += int(max(fps, 10)) * 8

        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps

        show_coordinates(frame, index)
        index += 1
        if index == n_coordinates:
            index = 0

        cv2.imshow("Camera Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("Total frames: " + str(total_frames))