#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
For testing rapid updates of ScatterPlotItem under various conditions.

(Scatter plots are still rather slow to draw; expect about 20fps)
"""



## Add path to library (just for examples; you do not need this)
#import initExample


#import PySide
from pyqtgraph.Qt import QtGui, QtCore, USE_PYSIDE
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.ptime import time
#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)
#USE_PYSIDE=True
if USE_PYSIDE:
    from ScatterPlotSpeedTestTemplate_pyside_gl import Ui_Form
else:
    from ScatterPlotSpeedTestTemplate_pyqt_gl import Ui_Form

win = QtGui.QWidget()
win.setWindowTitle('pyqtgraph example: ScatterPlotSpeedTest')
ui = Ui_Form()
ui.setupUi(win)
win.show()

p = ui.plot
N = 5e5

data = np.random.normal(size=(50,N,3))
data[:,:,2] = 0
sizeArray = (np.random.random(N) * 20.).astype(int)

ptr = 0
lastTime = time()
fps = None
def update():
    global curve, data, ptr, p, lastTime, fps
    #p.clear()
    if ui.randCheck.isChecked():
        size = sizeArray
    else:
        size = ui.sizeSpin.value() 
    pos = data[ptr%50]
    curve = gl.GLScatterPlotItem(pos=pos, 
            pxMode=True , color=(255,0,0,128), size=size)
    if len(p.items) > 0:
        for i in xrange(len(p.items)):
            p.removeItem(p.items[0])
    p.addItem(curve)
    ptr += 1
    now = time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    print '%0.2f fps' % fps
    p.update()
    #app.processEvents()  ## force complete redraw for every plot
p.opts['viewarea'] = [-30, 30]
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)
    


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
