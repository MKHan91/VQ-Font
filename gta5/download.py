import os

video_url = "https://www.youtube.com/watch?v=gvusDEF3eh4" 
save_path = "/home/dev/data/test_video.mp4"

os.system(f"yt-dlp -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' -o '{save_path}' {video_url}")
print("✅ 주행 영상 다운로드 완료!")