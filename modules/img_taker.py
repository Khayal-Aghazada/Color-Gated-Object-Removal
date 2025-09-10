# python -m modules.custom_train


from modules import Camera, ImageOps; import cv2, os, time
cam = Camera(0, mirror=False); os.makedirs("data/raw", exist_ok=True)
i=0
for f in cam.iterate():
    if cam.wait_key()==ord('s'):
        path=f"data/raw/{int(time.time())}_{i}.jpg"; ImageOps.save(path,f); i+=1
    cam.show("capture", f)
cam.release()



