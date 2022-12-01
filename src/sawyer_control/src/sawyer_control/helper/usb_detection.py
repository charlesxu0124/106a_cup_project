import pyudev
import rospy 
import threading
import logging as LOGGER

# source: https://vivekanandxyz.wordpress.com/2018/01/03/detecting-usb-insertion-removal-using-python/ 

class USBDetector():
    ''' Monitor udev for detection of usb '''
    
    def __init__(self):
        ''' Initiate the object '''

        self._reset()
        thread = threading.Thread(target=self._work)
        thread.daemon = True
        thread.start()
        

    def _work(self):
        ''' Runs the actual loop to detect the events '''
        self.context = pyudev.Context()
        self.monitor = pyudev.Monitor.from_netlink(self.context)
        self.monitor.filter_by(subsystem='usb')
        # this is module level logger, can be ignored
        LOGGER.info("Starting to monitor for usb")
        self.monitor.start()
        for device in iter(self.monitor.poll, None):
            LOGGER.info("Got USB event: %s", device.action)
            if device.action == 'add':
                # some function to run on insertion of usb
                self.inserted = True
            else:
                # some function to run on removal of usb
                self.inserted = False

    def _reset(self):
        self.inserted = False

if __name__ == "__main__":
    usb = USBDetector()
    while True:
        print(usb.inserted)
        rospy.sleep(0.05)