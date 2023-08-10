""" Script to detect laser points in the camera feed
    Modified version of Brad Montgomery's code
    https://github.com/bradmontgomery/python-laser-tracker/ """
import cv2
import sys
import numpy

# declare global variables
win_img = None
load_0 = None
load_1 = None
load_2= None
load_3 = None
count = 0

class LaserTracker(object):

    def __init__(self, cam_width=640, cam_height=480, hue_min=20, hue_max=160,
                 sat_min=100, sat_max=255, val_min=200, val_max=256):
        """
        * ``cam_width`` x ``cam_height`` -- This should be the size of the
        image coming from the camera. Default is 640x480.

        HSV color space Threshold values for a RED laser pointer are determined
        by:

        * ``hue_min``, ``hue_max`` -- Min/Max allowed Hue values
        * ``sat_min``, ``sat_max`` -- Min/Max allowed Saturation values
        * ``val_min``, ``val_max`` -- Min/Max allowed pixel values

        If the dot from the laser pointer doesn't fall within these values, it
        will be ignored.
        """

        self.cam_width = cam_width
        self.cam_height = cam_height
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.mask_num = 1
        self.win = 0
        
        self.capture = None  # camera capture device
        self.channels = {
            'hue': None,
            'saturation': None,
            'value': None,
            'laser': None,
            'key' : None
        }

        self.previous_position = None

    def create_and_position_window(self, name, scale, xpos, ypos):
        """Creates a named widow placing it on the screen at (xpos, ypos)."""
        # Create a window
        cv2.namedWindow(name)
        # Resize it to the size of the camera image
        if (scale):
            cv2.resizeWindow(name, self.cam_width*2, self.cam_height*2)
        else:
            cv2.resizeWindow(name, self.cam_width, self.cam_height)
        # Move to (xpos,ypos) on the screen
        cv2.moveWindow(name, xpos, ypos)

    def setup_camera_capture(self, device_num=0):
        """Perform camera setup for the device number (default device = 0).
        Returns a reference to the camera Capture object. """
        try:
            device = int(device_num)
            sys.stdout.write("Using Camera Device: {0}\n".format(device))
        except (IndexError, ValueError):
            # assume we want the 1st device
            device = 0
            sys.stderr.write("Invalid Device. Using default device 0\n")

        # Try to start capturing frames
        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            sys.stderr.write("Failed to Open Capture device. Quitting.\n")
            sys.exit(1)

        # set the wanted image size from the camera
        self.capture.set(
            cv2.cv.CV_CAP_PROP_FRAME_WIDTH if cv2.__version__.startswith('2') else cv2.CAP_PROP_FRAME_WIDTH,
            self.cam_width
        )
        self.capture.set(
            cv2.cv.CV_CAP_PROP_FRAME_HEIGHT if cv2.__version__.startswith('2') else cv2.CAP_PROP_FRAME_HEIGHT,
            self.cam_height
        )
        return self.capture

    def handle_quit(self, delay=10):
        """Quit the program if the user presses "Esc" or "q"."""
        key = cv2.waitKey(delay)
        c = chr(key & 255)
        if c in ['q', 'Q', chr(27)]:
            sys.exit(0)
        elif c in ['1']:
            self.mask_num = 1
            self.win = 0
        elif c in ['2']:
            self.mask_num = 2
            self.win = 0
        elif c in [' ', 'c', 'C']:
            self.win = 0
       

    def threshold_image(self, channel):
        if channel == "hue":
            minimum = self.hue_min
            maximum = self.hue_max
        elif channel == "saturation":
            minimum = self.sat_min
            maximum = self.sat_max
        elif channel == "value":
            minimum = self.val_min
            maximum = self.val_max

        (t, tmp) = cv2.threshold(
            self.channels[channel],  # src
            maximum,  # threshold value
            0,  # we dont care because of the selected type
            cv2.THRESH_TOZERO_INV  # t type
        )

        (t, self.channels[channel]) = cv2.threshold(
            tmp,  # src
            minimum,  # threshold value
            255,  # maxvalue
            cv2.THRESH_BINARY  # type
        )

        if channel == 'hue':
            # only works for filtering red color because the range for the hue
            # is split
            self.channels['hue'] = cv2.bitwise_not(self.channels['hue'])

    def display(self, frame):
        """Display the combined image and (optionally) all other image channels
        NOTE: default color space in OpenCV is BGR.
        """
        global win_img, load_0, load_1, load_2, load_3, count

        # cv2.imshow('RGB_VideoFrame', frame)
        cv2.imshow('Laser Light', self.channels['laser'])
        cv2.imshow('Key', self.channels['key'])
        # what to display in the indicating window
        if (self.win):
            cv2.imshow('Laser Key!', win_img)
        elif (count < 10):
            cv2.imshow('Laser Key!', load_0)
        elif (count <= 25):
            cv2.imshow('Laser Key!', load_1)
        elif (count <= 40):
            cv2.imshow('Laser Key!', load_2)
        else:
            cv2.imshow('Laser Key!', load_3)
        count = (count+1) % 55


    def setup_windows(self):
        sys.stdout.write("Using OpenCV version: {0}\n".format(cv2.__version__))
        global win_img, load_0, load_1, load_2, load_3

        # load in images to display for winning
        win_img = cv2.imread('imgs\win_img.png')
        load_0 = cv2.imread('imgs\load_0.png')
        load_1 = cv2.imread('imgs\load_1.png')
        load_2 = cv2.imread('imgs\load_2.png')
        load_3 = cv2.imread('imgs\load_3.png')

        # create output windows
        self.create_and_position_window('Laser Light', False,
                                         0, 0)
        # self.create_and_position_window('RGB_VideoFrame', False,
        #                                 10 + self.cam_width, 0)
        self.create_and_position_window('Key', False,
                                        10 + self.cam_width, 0) 
        self.create_and_position_window('Laser Key!', True,
                                        20 + self.cam_width, 10)

    def track(self, frame, mask):
        """
        Track the position of the laser pointer.

        Code taken from
        http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
        """
        center = None

        countours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]

        # only proceed if at least one contour was found
        if len(countours) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(countours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            moments = cv2.moments(c)
            if moments["m00"] > 0:
                center = int(moments["m10"] / moments["m00"]), \
                         int(moments["m01"] / moments["m00"])
            else:
                center = int(x), int(y)

        self.previous_position = center

    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # split the video frame into color channels
        h, s, v = cv2.split(hsv_img)
        self.channels['hue'] = h
        self.channels['saturation'] = s
        self.channels['value'] = v

        # Threshold ranges of HSV components; storing the results in place
        self.threshold_image("hue")
        self.threshold_image("saturation")
        self.threshold_image("value")

        # Perform an AND on HSV components to identify the laser!
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['hue'],
            self.channels['value']
        )
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['saturation'],
            self.channels['laser']
        )

        # track to find large enough point
        self.track(frame, self.channels['laser'])
        # display laser light identified in key boxes
        if (self.previous_position != None):
            # set win if pattern detected
            self.channels['key'] = cv2.bitwise_and(
                    self.channels['laser'], 
                    self.channels['key']
            )
            if (self.unlock() == 1):
                self.win = 1    

    def unlock(self):
        """ Analyse image for unlocking pattern """
        # set variables
        ox = 130
        oy = 130
        n = 0
        
        # select 'mask'
        if (self.mask_num == 1):
            mask = [[0, 0], [0, oy], [0, -oy], [-ox, 0], [ox, 0]]
        elif (self.mask_num == 2):
            mask = [[0, 0], [ox, 0], [-ox + 30, 0], [ox/2 - 50, 2*oy/3 - 10], [-ox/2, 2*oy/3], [ox/2, -2*oy/3 + 10], [-ox/2 + 30, -2*ox/3 + 30]]

        passes = numpy.zeros(len(mask))

        # define centre point
        c_x = self.previous_position[0] #int(self.cam_width / 2)
        c_y = self.previous_position[1] #int(self.cam_height / 2)

        # check for points around centre
        for offset in mask: 
            flag = 0
            sq_count = 0
            for x in range(-20, 20):
                if (flag):
                    break
                for y in range(-20, 20):
                    x_pos = x + c_x + int(offset[0])
                    y_pos = y + c_y + int(offset[1])
                    # keep it all in bounds - for variable centre only
                    if (x_pos < 0):
                        x_pos = 0
                    else:
                        x_pos = min(self.cam_width-1, x_pos)
                    if (y_pos < 0):
                        y_pos = 0
                    else:
                        y_pos = min(self.cam_height-1, y_pos)
                    # weigth the spots detected
                    self.channels['key'][y_pos, x_pos] = 255
                    if (self.channels['laser'][y_pos, x_pos]):
                        sq_count += 1
                        if (sq_count >= 5):
                            passes[n] = 1
                            flag = 1
                            break
            # increase count`
            n += 1
        
        # return bool - i.e. whether we've found something in every square 
        return not(0 in passes)
        
    def run(self):
        # Set up window positions
        self.setup_windows()
        # Set up the camera capture from device 2 - Phone over USB
        self.setup_camera_capture(2)

        # set up mask and key channels
        self.channels['key'] = numpy.zeros((self.cam_height, self.cam_width, 1),
                                    numpy.uint8)
        
        for n in range(5):
            success, frame = self.capture.read()
            if not success:  # no image captured... end the processing
                sys.stderr.write("Could not read camera frame. Quitting\n")
                sys.exit(1)

        while True:
            # 1. capture the current image
            success, frame = self.capture.read()
            if not success:  # no image captured... end the processing
                sys.stderr.write("Could not read camera frame. Quitting\n")
                sys.exit(1)

            self.detect(frame)
            self.display(frame)
            self.handle_quit()

if __name__ == '__main__':
    """ Change the '...min' / '...max' values to change the 
        sensitivity of the tracker to laser light """
    tracker = LaserTracker(
        cam_width=640,
        cam_height=480,
        hue_min=0,
        hue_max=60,
        sat_min=30,
        sat_max=255,
        val_min=140,
        val_max=255
    )
    tracker.run()