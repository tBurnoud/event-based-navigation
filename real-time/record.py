import dv_processing as dv
import argparse
from os.path import exists

parser = argparse.ArgumentParser(description='Show a preview of an AEDAT4 recording.')

parser.add_argument('-f,--file',
                    dest='file',
                    type=str,
                    required=True,
                    metavar='path/to/file',
                    help='Path to an AEDAT4 file')

args = parser.parse_args()

if __name__ == "__main__":
    filePath = "../" + args.file + ".aedat4"
    if not exists(filePath):
        camera = dv.io.CameraCapture()
        config = dv.io.MonoCameraWriter.EventOnlyConfig("DXB00078", camera.getEventResolution())
        print("start recording...")
        try:
            writer = dv.io.MonoCameraWriter(filePath, config)
            while True:
                events = camera.getNextEventBatch()
                if events is not None:
                    writer.writeEvents(events)
        except KeyboardInterrupt:
            pass
        
        print("end of recording")
    else:
        print("file does already exist, please find another name for the recording")