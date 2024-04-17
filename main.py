from utils import (read_video, save_video)
from tracker import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
from mini_court import MiniCourt

def main():
    input_video_path= "videos\input_video.mp4"
    video_frames = read_video(input_video_path)
    ball_tracker= BallTracker(model_path="model\yolo5_last.pt")
    player_tracker=PlayerTracker(model_path='yolov8x.pt')
    player_detections= player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detection.pkl")
    ball_detections= ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detection.pkl")
    ball_detections= ball_tracker.interpolate_ball_positions(ball_detections)

    court_model_path= "model\keypoints_model.pth"
    court_line_detector= CourtLineDetector(court_model_path)
    court_keypoints= court_line_detector.predict(video_frames[0])
    player_detections= player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    mini_court= MiniCourt(video_frames[0])

    output_video_frame= player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frame= ball_tracker.draw_bboxes(output_video_frame, ball_detections)
    output_video_frame= court_line_detector.draw_keypoints_on_video(output_video_frame, court_keypoints)

    output_video_frame= mini_court.draw_mini_court(output_video_frame)

    for i, frame in enumerate(output_video_frame):
        cv2.putText(frame, f'Frame: {i}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,225,0), 2)
    save_video(output_video_frame,"output_videos/output_video.avi")
    

if __name__ == "__main__":
    main()