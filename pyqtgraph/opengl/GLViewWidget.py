from pyqtgraph.Qt import QtCore, QtGui, QtOpenGL
from OpenGL.GL import *
import OpenGL.GL.framebufferobjects as glfbo
import numpy as np
from pyqtgraph import Vector
import pyqtgraph.functions as fn

##Vector = QtGui.QVector3D



class GLCamera3D(object):
    def __init__(self, distance=10., fov=60, elevation=30, azimuth=45):
        super(GLCamera3D, self).__init__()
        self.distance = distance ## distance of camera from center
        self.fov = fov ## horizontal field of view in degrees
        self.elevation = elevation ## camera's angle of elevation in degrees
        self.azimuth = azimuth ## (rotation around z-axis 0 points along x-axis)

    def _getDistance(self):
        return self._distance

    def _setDistance(self, distance):
        self._distance = distance
    distance = property(_getDistance, _setDistance)

    def _getElevation(self):
        return self._elevation

    def _setElevation(self, elevation):
        self._elevation = np.clip(elevation, -90, 90)
    elevation = property(_getElevation, _setElevation)

    def _getAzimuth(self):
        return self._azimuth

    def _setAzimuth(self, azimuth):
        self._azimuth = azimuth
    property(_getAzimuth, _setAzimuth)
     
    def _getFOV(self):
        return self._fov
    
    def _setFOV(self, fov):
        self._fov = fov

    def setPosition(self, pos=None, distance=None, elevation=None, azimuth=None):
        if distance is not None:
            self.distance = distance
        if elevation is not None:
            self.elevation = elevation
        if azimuth is not None:
            self.azimuth = azimuth

    def getPosition(self, center):
        """Return current position of camera based on center, dist, elevation, and azimuth"""
        dist = self.distance
        elev = self.elevation * np.pi/180.
        azim = self.azimuth * np.pi/180.
        
        pos = Vector(
            center.x() + dist * np.cos(elev) * np.cos(azim),
            center.y() + dist * np.cos(elev) * np.sin(azim),
            center.z() + dist * np.sin(elev)
        )
        return pos


class GLCamera2D(GLCamera3D):
    def __init__(self, distance=10., fov=60, elevation=90, azimuth=90):
        super(GLCamera2D, self).__init__(distance, fov, elevation,
                azimuth)

    def _setElevation(self, elevation):
        pass #raise Warning("2D Camera does not allow setting elevation, use 3D camera instead")

    def _getElevation(self):
        return 90
    elevation = property(_getElevation, _setElevation)

 


class GLViewWidget(QtOpenGL.QGLWidget):
    """
    Basic widget for displaying 3D data
        - Rotation/scale controls
        - Axis/grid display
        - Export options

    """
    
    ShareWidget = None
    
    def __init__(self, parent=None, camera = None):
        if GLViewWidget.ShareWidget is None:
            ## create a dummy widget to allow sharing objects (textures, shaders, etc) between views
            GLViewWidget.ShareWidget = QtOpenGL.QGLWidget()
            
        #QtOpenGL.QGLWidget.__init__(self, parent, GLViewWidget.ShareWidget)
        super(GLViewWidget, self).__init__(parent, GLViewWidget.ShareWidget)

        self.opts = {
            'center': Vector(0,0,0),  ## will always appear at the center of the widget self.setFocusPolicy(QtCore.Qt.ClickFocus)
            'viewport': None}         ## glViewport params; None == whole widget
        
        self.items = []
        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown]
        self.keysPressed = {}
        self.keyTimer = QtCore.QTimer()
        self.keyTimer.timeout.connect(self.evalKeyState)
        if not camera:
            self.camera = GLCamera3D()
        else:
            self.camera = camera
        
        self.makeCurrent()

    def cameraPosition(self):
        return self.camera.getPosition(self.opts['center'])

    def setCameraPosition(self, pos=None, distance=None, elevation=None, azimuth=None):
        self.camera.setPosition(pos, distance, elevation, azimuth)
        self.update()

    def pan(self, dx, dy, dz, relative=False):
        """
        Moves the center (look-at) position while holding the camera in place. 
        
        If relative=True, then the coordinates are interpreted such that x
        if in the global xy plane and points to the right side of the view, y is
        in the global xy plane and orthogonal to x, and z points in the global z
        direction. Distances are scaled roughly such that a value of 1.0 moves
        by one pixel on screen.
        
        """
        if not relative:
            self.opts['center'] += QtGui.QVector3D(dx, dy, dz)
        else:
            cPos = self.camera.getPosition(self.opts['center'])
            cVec = self.opts['center'] - cPos
            dist = cVec.length()  ## distance from camera to center
            fov = self.camera.fov
            xDist = dist * 2. * np.tan(0.5 * fov * np.pi / 180.)  ## approx. width of view at distance of center point
            xScale = xDist / self.width()
            zVec = QtGui.QVector3D(0,0,1)
            if self.camera.elevation == 90:
                xVec = QtGui.QVector3D(1,0,0)
                yVec = QtGui.QVector3D(0,1,0)
            else:
                xVec = QtGui.QVector3D.crossProduct(zVec, cVec).normalized()
                yVec = QtGui.QVector3D.crossProduct(xVec, zVec).normalized()
            self.opts['center'] = self.opts['center'] + xVec * xScale * dx + yVec * xScale * dy + zVec * xScale * dz
        self.update()

    def orbit(self, azim, elev):
        """Orbits the camera around the center position. *azim* and *elev* are given in degrees."""
        self.camera.azimuth += azim
        #self.opts['elevation'] += elev
        self.camera.elevation = self.camera.elevation + elev
        print self.camera.elevation
        self.update()

    def pixelSize(self, pos):
        """
        Return the approximate size of a screen pixel at the location pos
        Pos may be a Vector or an (N,3) array of locations
        """
        cam = self.camera.getPosition(self.opts['center'])
        if isinstance(pos, np.ndarray):
            cam = np.array(cam).reshape((1,)*(pos.ndim-1)+(3,))
            dist = ((pos-cam)**2).sum(axis=-1)**0.5
        else:
            dist = (pos-cam).length()
        xDist = dist * 2. * np.tan(0.5 * self.camera.fov * np.pi / 180.)
        return xDist / self.width()

    def zoom(self, delta, mod=False):
        if mod:
            self.camera.fov *= 0.999**delta
        else:
            self.camera.distance *= 0.999**delta
        self.update()

    def setProjection(self, region=None):
        m = self.projectionMatrix(region)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        a = np.array(m.copyDataTo()).reshape((4,4))
        glMultMatrixf(a.transpose())

    def projectionMatrix(self, region=None):
        # Xw = (Xnd + 1) * width/2 + X
        if region is None:
            region = (0, 0, self.width(), self.height())

        x0, y0, w, h = self.getViewport()
        dist = self.camera.distance
        fov = self.camera.fov
        nearClip = dist * 0.001
        farClip = dist * 1000.

        r = nearClip * np.tan(fov * 0.5 * np.pi / 180.)
        t = r * h / w

        # convert screen coordinates (region) to normalized device coordinates
        # Xnd = (Xw - X0) * 2/width - 1
        ## Note that X0 and width in these equations must be the values used in viewport
        left  = r * ((region[0]-x0) * (2.0/w) - 1)
        right = r * ((region[0]+region[2]-x0) * (2.0/w) - 1)
        bottom = t * ((region[1]-y0) * (2.0/h) - 1)
        top    = t * ((region[1]+region[3]-y0) * (2.0/h) - 1)

        tr = QtGui.QMatrix4x4()
        tr.frustum(left, right, bottom, top, nearClip, farClip)
        return tr

    def setModelview(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        m = self.viewMatrix()
        a = np.array(m.copyDataTo()).reshape((4,4))
        glMultMatrixf(a.transpose())

    def viewMatrix(self):
        tr = QtGui.QMatrix4x4()
        tr.translate( 0.0, 0.0, -self.camera.distance)
        tr.rotate(self.camera.elevation-90, 1, 0, 0)
        tr.rotate(self.camera.azimuth+90, 0, 0, -1)
        center = self.opts['center']
        tr.translate(-center.x(), -center.y(), -center.z())
        return tr
     
    def addItem(self, item):
        self.items.append(item)
        if hasattr(item, 'initializeGL'):
            self.makeCurrent()
            try:
                item.initializeGL()
            except:
                self.checkOpenGLVersion('Error while adding item %s to GLViewWidget.' % str(item))
                
        item._setView(self)
        #print "set view", item, self, item.view()
        self.update()
        
    def removeItem(self, item):
        self.items.remove(item)
        item._setView(None)
        self.update()
        
        
    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        self.resizeGL(self.width(), self.height())
        
       
    def resizeGL(self, w, h):
        pass
        #glViewport(*self.getViewport())
        #self.update()

    def itemsAt(self, region=None):
        #buf = np.zeros(100000, dtype=np.uint)
        buf = glSelectBuffer(100000)
        try:
            glRenderMode(GL_SELECT)
            glInitNames()
            glPushName(0)
            self._itemNames = {}
            self.paintGL(region=region, useItemNames=True)
            
        finally:
            hits = glRenderMode(GL_RENDER)

        items = [(h.near, h.names[0]) for h in hits]
        items.sort(key=lambda i: i[0])
        
        return [self._itemNames[i[1]] for i in items]
        
    def paintGL(self, region=None, viewport=None, useItemNames=False):
        """
        viewport specifies the arguments to glViewport. If None, then we use self.opts['viewport']
        region specifies the sub-region of self.opts['viewport'] that should be rendered.
        Note that we may use viewport != self.opts['viewport'] when exporting.
        """
        if viewport is None:
            glViewport(*self.getViewport())
        else:
            glViewport(*viewport)
        self.setProjection(region=region)
        self.setModelview()
        glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT )
        self.drawItemTree(useItemNames=useItemNames)
        
    def drawItemTree(self, item=None, useItemNames=False):
        if item is None:
            items = [x for x in self.items if x.parentItem() is None]
        else:
            items = item.childItems()
            items.append(item)
        items.sort(key=lambda a: a.depthValue())
        for i in items:
            if not i.visible():
                continue
            if i is item:
                try:
                    glPushAttrib(GL_ALL_ATTRIB_BITS)
                    if useItemNames:
                        glLoadName(id(i))
                        self._itemNames[id(i)] = i
                    i.paint()
                except:
                    import pyqtgraph.debug
                    pyqtgraph.debug.printExc()
                    msg = "Error while drawing item %s." % str(item)
                    ver = glGetString(GL_VERSION)
                    if ver is not None:
                        ver = ver.split()[0]
                        if int(ver.split(b'.')[0]) < 2:
                            print(msg + " The original exception is printed above; however, pyqtgraph requires OpenGL version 2.0 or greater for many of its 3D features and your OpenGL version is %s. Installing updated display drivers may resolve this issue." % ver)
                        else:
                            print(msg)
                    
                finally:
                    glPopAttrib()
            else:
                glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                try:
                    tr = i.transform()
                    a = np.array(tr.copyDataTo()).reshape((4,4))
                    glMultMatrixf(a.transpose())
                    self.drawItemTree(i, useItemNames=useItemNames)
                finally:
                    glMatrixMode(GL_MODELVIEW)
                    glPopMatrix()

      
       
    def mousePressEvent(self, ev):
        self.mousePos = ev.pos()
        
    def mouseMoveEvent(self, ev):
        diff = ev.pos() - self.mousePos
        self.mousePos = ev.pos()
        
        if ev.buttons() == QtCore.Qt.LeftButton:
            self.orbit(-diff.x(), diff.y())
            #print self.opts['azimuth'], self.opts['elevation']
        elif ev.buttons() == QtCore.Qt.RightButton:
            if (ev.modifiers() & QtCore.Qt.ControlModifier):
                self.pan(diff.x(), 0, diff.y(), relative=True)
            else:
                self.pan(diff.x(), diff.y(), 0, relative=True)
        
    def mouseReleaseEvent(self, ev):
        pass
        
    def wheelEvent(self, ev):
        if (ev.modifiers() & QtCore.Qt.ControlModifier):
            self.zoom(ev.delta(), mod=True)
        else:
            self.zoom(ev.delta())

    def keyPressEvent(self, ev):
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
      
    def keyReleaseEvent(self, ev):
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except:
                self.keysPressed = {}
            self.evalKeyState()
        
    def evalKeyState(self):
        speed = 2.0
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == QtCore.Qt.Key_Right:
                    self.orbit(azim=-speed, elev=0)
                elif key == QtCore.Qt.Key_Left:
                    self.orbit(azim=speed, elev=0)
                elif key == QtCore.Qt.Key_Up:
                    self.orbit(azim=0, elev=-speed)
                elif key == QtCore.Qt.Key_Down:
                    self.orbit(azim=0, elev=speed)
                elif key == QtCore.Qt.Key_PageUp:
                    pass
                elif key == QtCore.Qt.Key_PageDown:
                    pass
                self.keyTimer.start(16)
        else:
            self.keyTimer.stop()

    def checkOpenGLVersion(self, msg):
        ## Only to be called from within exception handler.
        ver = glGetString(GL_VERSION).split()[0]
        if int(ver.split('.')[0]) < 2:
            import pyqtgraph.debug
            pyqtgraph.debug.printExc()
            raise Exception(msg + " The original exception is printed above; however, pyqtgraph requires OpenGL version 2.0 or greater for many of its 3D features and your OpenGL version is %s. Installing updated display drivers may resolve this issue." % ver)
        else:
            raise
            

            
    def readQImage(self):
        """
        Read the current buffer pixels out as a QImage.
        """
        w = self.width()
        h = self.height()
        self.repaint()
        pixels = np.empty((h, w, 4), dtype=np.ubyte)
        pixels[:] = 128
        pixels[...,0] = 50
        pixels[...,3] = 255
        
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
        
        # swap B,R channels for Qt
        tmp = pixels[...,0].copy()
        pixels[...,0] = pixels[...,2]
        pixels[...,2] = tmp
        pixels = pixels[::-1] # flip vertical
        
        img = fn.makeQImage(pixels, transpose=False)
        return img
        
        
    def renderToArray(self, size, format=GL_BGRA, type=GL_UNSIGNED_BYTE, textureSize=1024, padding=256):
        w,h = map(int, size)
        
        self.makeCurrent()
        tex = None
        fb = None
        try:
            output = np.empty((w, h, 4), dtype=np.ubyte)
            fb = glfbo.glGenFramebuffers(1)
            glfbo.glBindFramebuffer(glfbo.GL_FRAMEBUFFER, fb )
            
            glEnable(GL_TEXTURE_2D)
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            texwidth = textureSize
            data = np.zeros((texwidth,texwidth,4), dtype=np.ubyte)
            
            ## Test texture dimensions first
            glTexImage2D(GL_PROXY_TEXTURE_2D, 0, GL_RGBA, texwidth, texwidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 0:
                raise Exception("OpenGL failed to create 2D texture (%dx%d); too large for this hardware." % shape[:2])
            ## create teture
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texwidth, texwidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.transpose((1,0,2)))
            
            self.setViewport((0, 0, w, h))  # viewport is the complete image; this ensures that paintGL(region=...) 
                                                  # is interpreted correctly.
            p2 = 2 * padding
            for x in range(-padding, w-padding, texwidth-p2):
                for y in range(-padding, h-padding, texwidth-p2):
                    x2 = min(x+texwidth, w+padding)
                    y2 = min(y+texwidth, h+padding)
                    w2 = x2-x
                    h2 = y2-y
                    
                    ## render to texture
                    glfbo.glFramebufferTexture2D(glfbo.GL_FRAMEBUFFER, glfbo.GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
                    
                    self.paintGL(region=(x, h-y-h2, w2, h2), viewport=(0, 0, w2, h2))  # only render sub-region
                    
                    ## read texture back to array
                    data = glGetTexImage(GL_TEXTURE_2D, 0, format, type)
                    data = np.fromstring(data, dtype=np.ubyte).reshape(texwidth,texwidth,4).transpose(1,0,2)[:, ::-1]
                    output[x+padding:x2-padding, y+padding:y2-padding] = data[padding:w2-padding, -(h2-padding):-padding]
                    
        finally:
            self.setViewport(None)
            glfbo.glBindFramebuffer(glfbo.GL_FRAMEBUFFER, 0)
            glBindTexture(GL_TEXTURE_2D, 0)
            if tex is not None:
                glDeleteTextures([tex])
            if fb is not None:
                glfbo.glDeleteFramebuffers([fb])
            
        return output
        

    def getViewport(self):
        vp = self.opts['viewport']
        if vp is None:
            return (0, 0, self.width(), self.height())
        else:
            return vp

    def setViewport(self, viewport):
        self.opts['viewport'] = viewport
     

class GLViewWidget2D(GLViewWidget):
    def __init__(self, parent=None):
        camera = GLCamera2D()
        super(GLViewWidget2D, self).__init__(parent, camera=camera)

    def evalKeyState(self):
        speed = 1.0
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == QtCore.Qt.Key_Right:
                    self.pan(+speed, 0)
                elif key == QtCore.Qt.Key_Left:
                    self.pan(-speed, 0)
                elif key == QtCore.Qt.Key_Up:
                    self.pan(0, speed)
                elif key == QtCore.Qt.Key_Down:
                    self.pan(0, -speed)
                elif key == QtCore.Qt.Key_PageUp:
                    self.orbit(4*speed)
                elif key == QtCore.Qt.Key_PageDown:
                    self.orbit(-4*speed)
                self.keyTimer.start(16)
        else:
            self.keyTimer.stop()

    def mouseMoveEvent(self, ev):
        diff = ev.pos() - self.mousePos
        self.mousePos = ev.pos()
        
        if ev.buttons() == QtCore.Qt.RightButton:
            self.orbit(diff.x(), diff.y())
        elif ev.buttons() == QtCore.Qt.LeftButton:
            if (ev.modifiers() & QtCore.Qt.ControlModifier):
                self.pan(-diff.x(), -diff.y(), 0, relative=False)
            else:
                self.pan(diff.x(), -diff.y(), 0, relative=True)
