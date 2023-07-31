import os
import time
import streamlit as st
import cv2

from predict import build_model, detect, format_yolov5, wrap_detection
import sys
from detect import detect as yolov7detect
from pathlib import Path
import uuid
from download import download_button


def main():
    class_list = ["pothole"]
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
    st.set_page_config(page_title="Road Warrior")
    st.title("Road Warrior")
    st.caption("Powered by OpenCV, Streamlit & Sagemaker")
    capture = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")
    st.session_state.disable_inference = False
    st.session_state.button_text = "Analyze"

    with st.form(key="Form :", clear_on_submit = True):
        file_uploaded = st.file_uploader(label = "Upload Pic", type=["png", "jpeg", "webp", "mp4"])
        submit_clicked = st.form_submit_button(label=st.session_state.button_text, disabled=st.session_state.get("disable_inference", False))

    if submit_clicked and file_uploaded is not None:
        save_folder = "input/"
        save_path = Path(save_folder, file_uploaded.name)
        with open(save_path, mode='wb') as w:
            w.write(file_uploaded.getvalue())

        mimetype = "video/mp4"
        mimetype = "image/png" if "png" in file_uploaded.name or "jpg" in file_uploaded.name or "jpeg" in file_uploaded.name or "webp" in file_uploaded.name else mimetype
        
        if save_path.exists():
            st.session_state.disable_inference = True
            st.session_state.button_text = "Analysing..."
            # output_filepath = yolov7detect(source=save_path.as_posix(), weights=os.path.join(os.getcwd(), 'models', 'best.pt'))
            capture = cv2.VideoCapture(save_path.as_posix())
            st.session_state.disable_inference = False
            st.session_state.button_text = "Analyze"
            # if output_filepath is not None:
            #     st.success('Inference successful!')
            #     with open(output_filepath, 'rb') as f:
            #         download_button_str = download_button(f.read(), output_filepath, mimetype, "Download Output")
            #         st.markdown(download_button_str, unsafe_allow_html=True)

            # else:
            #     st.error("Inference failed")

    net = build_model(is_cuda)

    start = time.time_ns()
    frame_count = 0
    total_frames = 0
    fps = -1

    while True:
        _, frame = capture.read()
        if frame is None:
            print("End of stream")
            break

        inputImage = format_yolov5(frame)
        outs = detect(inputImage, net)

        class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

        frame_count += 1
        total_frames += 1

        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()
        
        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame,channels="RGB")
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break

    print("Total frames: " + str(total_frames))

if __name__ == "__main__":
    main()

