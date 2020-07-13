import argparse
import time
import cv2
import numpy as np
import sys
import os

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from vision import analyze, workouts, helpers


fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='sample_curls_2.mov')
    parser.add_argument('--resize', type=str, default='160x160', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--workout', type=str, default="curls",
                        help='shoulderpress, plank, curls, squats, pushup')
    parser.add_argument('--side', type=str, default="L", help='L for left or R for right')
    parser.add_argument('--setrep', nargs='+', type=int, default=[10, 15], help='Sets and respective reps') #e.g. --setrep 10 8 6 --> args.setrep = [10, 8 ,6]
    parser.add_argument('--output', type=str, default="output_file_v4-nobicep_ct.avi",help='A file or directory to save output visualizations. If directory doesn\'t exist, it will be created.')
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    assert os.path.isfile(args.video)

    cap = cv2.VideoCapture(args.video)
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.output:
        basename = os.path.basename(args.video)
        abspath = os.path.join(os.path.abspath('.'), args.output)

        if os.path.splitext(abspath)[1] == '': #abspath specifies a directory which may or may not exist
            if not os.path.exists(abspath):
                os.makedirs(abspath)
            output_fname = os.path.join(abspath, basename)
            output_fname = os.path.splitext(output_fname)[0] + '_critique.avi'

        else: #abspath specifies a file
            if not os.path.exists(os.path.split(abspath)[0]):
                os.makedirs(os.path.split(abspath)[0])
            output_fname = abspath

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_fname, fourcc, fps = int(frames_per_second), frameSize = (cap_width, cap_height), isColor = True)

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    if args.workout == "shoulderpress":
        WorkOut = workouts.ShoulderPress(args.setrep, w, h)
    elif args.workout == "curls":
        WorkOut = workouts.Curls(args.setrep, w, h,args.side)


    check_specific_critique = ""
    prev_rep = 0
    count = 1 
    count_error = 0 
    count_minusone = 0
    while cap.isOpened():
        # print("Number of Frame : %d" %count)
        count = count + 1
        ret_val, image = cap.read()
        if not ret_val:
            break

        try:
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
            body_parts = analyze.extract_body_parts(humans, image, w, h)
            if body_parts == -1 :
                count_minusone = count_minusone+1
            bool, critique, func = WorkOut.run_critique(body_parts, check_specific_critique)
            WorkOut.update_state()
            # bool, critique, func = shoulderpress.run_critique(body_parts, check_specific_critique)
        except:
            # print("Number of Error : %d" %count_error)
            count_error = count_error +1 
            continue

        if bool:
            prev_rep = WorkOut.reps
            check_specific_critique = func

        if WorkOut.reps != prev_rep:
            check_specific_critique = ""

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        cv2.putText(image,"FPS: %f" %0, (10, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        cv2.putText(image,"Workout: %s" %args.workout ,(10, 45),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
        cv2.putText(image,"Critique: %s" %critique ,(10, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        cv2.putText(image, "Sets: %d" %WorkOut.sets, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        cv2.putText(image, "Reps: %d" %WorkOut.reps, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        # debugging only if not need comment the next 3 line out
        cv2.putText(image, "Shoulder_angle: %.2f" %WorkOut.shoulder_angle, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 0, 255), 2)
        cv2.putText(image, "Elbow_angle: %.2f" %WorkOut.elbow_angle, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 0, 255), 2)
        cv2.putText(image, "writst_pos = %s,elbow_pos = %s, shoulder_pos = %s" %(WorkOut.wrist_pos, WorkOut.elbow_pos, WorkOut.shoulder_pos), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 0, 255), 2)
        fps_time = time.time()

        if cv2.waitKey(1) == 25:
            break

        if args.output:
            writer.write(image)
        else:
            cv2.imshow('SimpL', image)

    print("Number of Frame : %d" %count)
    print("Number of Error : %d" %count_error)
    cv2.destroyAllWindows()
    cap.release()
    if args.output:
        writer.release()
