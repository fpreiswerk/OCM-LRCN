import cv2
# import os
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def render_movie(frames, output, framerate, text=False):

    height = frames.shape[1]
    width = frames.shape[2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = 0.6
    linewidth = 0

    # # build a lookup table mapping the pixel values [0, 255] to
    # # their adjusted gamma values
    # gamma = 2.0
    # invGamma = 1.0 / gamma
    # table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    if text and len(text) > 20 and len(text) < 40:
        textsize = 0.45

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    # fourcc = cv2.VideoWriter_fourcc(*'avcl') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, framerate, (width, height))

    for i in range(0, frames.shape[0]):
        frame = np.transpose(np.squeeze(frames[i, :, :]))
        frame = np.reshape(frame, (frame.shape[0], frame.shape[1], 1))

        frame = cv2.cvtColor(np.uint8(frame*255), cv2.COLOR_GRAY2RGB)
        # # apply gamma correction using the lookup table
        # frame = cv2.LUT(frame, table)

        if text:
            cv2.putText(frame, text, (10, 178), font, textsize, (255,255,255),
                        linewidth, cv2.LINE_4) #cv2.LINE_AA)
        out.write(frame)  # Write out frame to video

    # Release everything when job is finished
    out.release()
    cv2.destroyAllWindows()
    print("Video saved as {}".format(output))


def play_movie(frames, framerate):

    height = frames.shape[1]
    width = frames.shape[2]

    for i in range(0, frames.shape[0]):
        frame = np.transpose(np.squeeze(frames[i, :, :]))
        frame = np.reshape(frame, (frame.shape[0], frame.shape[1], 1))
        frame = cv2.cvtColor(np.uint8(frame*255), cv2.COLOR_GRAY2RGB)
        cv2.imshow('video', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
