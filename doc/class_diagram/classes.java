import java.util.Map;

/**
* @opt hide java.*
* @opt operations
* @opt attributes
* @opt types
* @hidden
*/
class UMLOptions {}

/** @hidden */
class Vec4f {}
/** @hidden */
class Vector3f {}
/** @hidden */
class Vec3Array {}
/** @hidden */
class Vec4Array {}
/** @hidden */
class BoundingSphere {}
/** @hidden */
class DrawElementsUInt {}
/** @hidden */
class Pair<A, B> {}
/** @hidden */
class Matrix4f {}
/** @hidden */
class Hidden {}
/** @hidden */
class Orientation {}
/** @hidden */
class Any {}

/** @hidden */
class RecordingParams
{
    public float simulationStart;
    public float simulationEnd;
    public float simulationDelta;
    public float cameraPathDelta;
    public CameraPath cameraPath;
    public String filePrefix;
    public String fileFormat;
    public String outputPath;

    boolean stopAtCameraPathEnd;
};

/** @hidden */
class AttributeMap
{
}

/**
   @shape note
*/
class EqualizerConfig
{
}

/**
   @composed 1 "" * View
   @has * <weak-ref> 1 Scene
   @composed 1 "" 1 SimulationPlayer
   @depend - <manages> - EqualizerConfig
*/
class RTNeuron
{
    public RTNeuron(int argc, char argv[], AttributeMap attributes);
    public void init(String configFileName);
    public void exit();
    public View[] getViews();
    public Scene createScene(AttributeMap attributes);
    public SimulationPlayer getPlayer();
    public void record(RecordingParams params);
    public void pause();
    public void resume();
    public void frame();
    public void waitFrame();
    public void waitFrames(int frames);
    public void wait();
    public void waitRecord();
    public AttributeMap getAttributes();
    public static String getVersionString();

    /** @hidden */
    public class FrameIssuedSignal {}

    FrameIssuedSignal frameIssued;
}

class Camera
{
    public void setProjectionPerspective(float verticalFOV);
    public void makeOrtho();
    public void makePerspective();
    public void setProjectionFrustum(float left, float right,
                                     float top, float bottom,
                                     float near, float far);
    public void setProjectionOrtho(float left, float right,
                                   float top, float bottom,
                                   float near, float far);
    public void setViewLookAt(Vector3f eye, Vector3f center, Vector3f up);
    public void setView(Vector3f position, Orientation orientation);
    public void getView(Vector3f position, Orientation orientation);
}

/**
   @has 1 "" 1 Camera
   @has * "displays" 0..1 Scene
   @has * "" 1 ColorMap
   @has * "" 1 Pointer
   @has * "" 1 CameraManipulator
 */
class View
{
    public AttributeMap getAttributes();
    public void record(boolean enable);
    public void snapshot(String filename, boolean waitForCompletion);
}

class Neurons
{
}

class Synapses
{
}

class CompartmentReportReader
{
}

/**
   @has * "" * Neurons
   @has * "" * Synapses
   @has 1 "" * Scene.Object
   @has * "" 0..1 SpikeData
   @has * "" 0..1 CompartmentReportReader
*/
class Scene
{
    public abstract class Object
    {
        public abstract AttributeMap getAttributes();
        public abstract void update();
        public abstract Any getObject();
    }

    public AttributeMap getAttributes();
    public Object addNeurons(Neurons neurons, AttributeMap attributes);
    public Object addEfferentSynapses(Synapses synapses,
                                      AttributeMap attributes);
    public Object addAfferentSynapses(Synapses synapses,
                                      AttributeMap attributes);
    public Object addModel(String filename, Matrix4f transform);
    public Object addModel(String filename, String transform);
    public Object addMesh(Vec3Array vertices,
                          DrawElementsUInt primitive,
                          Vec4Array colors,
                          Vec3Array normals,
                          AttributeMap attributes);
    public void update();
    public Object[] getObjects();
    public void remove(Object object);
    public void clear();
    public void highlight(int cell, boolean on);
    public void setSimulation(CompartmentReportReader reader);
    public void setSimulation(SpikeData spikeData);
    public BoundingSphere getCircuitSceneBoundingSphere();

    /** @hidden */
    public class CellSelectedSignal {}
    /** @hidden */
    public class ModelSelectedSignal {}

    public CellSelectedSignal cellSelected;
    public ModelSelectedSignal modelSelected;
}

class ColorMap
{
    //public void setPoints(Map<float, Vec4f> colorPoints);
    //public Map<float, Vec4f> getPoints();
    public void setPoints(Map colorPoints);
    public Map getPoints();
    public void setRange(float min, float max);
    public Vec4f getColor(float value);
    public void load(String fileName);
    public void save(String fileName);
    public void setTextureSize(int texels);
    public int getTextureSize();
}

class SpikeData
{
    public void load(String filename);
}

class SimulationPlayer
{
    public void setTimestamp(float milliseconds);
    public void setBeginTime(float milliseconds);
    public void setEndTime(float milliseconds);
    public void setSimulationDelta(float milliseconds);
    public void play();
    public void pause();

    /** @hidden */
    public class TimestampChangedSignal {}
    /** @hidden */
    public class PlaybackFinishedSignal {}

    public TimestampChangedSignal timestampChanged;
    public PlaybackFinishedSignal finished;
}

/**
   @composed 0..1 "" * CameraPath.KeyFrame
 */
class CameraPath
{
    /** @hidden */
    public class KeyFrame
    {
        public Vector3f position;
        public Orientation orientation;
        public float stereoCorrection;
    }

    public double getStartTime();
    public double getStopTime();
    //public void setKeyFrames(Map<double, KeyFrame> frames);
    public void setKeyFrames(Map frames);
    public void addKeyFrame(double seconds, KeyFrame frame);
    public void addKeyFrame(double seconds, View view);
    public void replaceKeyFrame(int index, KeyFrame frame);
    public KeyFrame getKeyFrame(int index);
    public void removeKeyFrame(int index);
    //public Pair<double, KeyFrame>[] getKeyFrames();
    Pair[] getKeyFrames();
    public void clear();
    public void load(String fileName);
    public void save(String fileName);
}

class CameraManipulator
{
}

/**
   @has * "" 1 CameraPath
*/
class CameraPathManipulator extends CameraManipulator
{
    public enum LoopMode {LOOP_NONE, LOOP_REPEAT, LOOP_SWING}
    public void load(String fileName);
    public void setPath(CameraPath cameraPath);
    public void setPlaybackStart(float start);
    public void setPlaybackStop(float end);
    public void setFrameDelta(float milliseconds);
    public void setLoopMode(LoopMode loopMode);
}

class TrackballManipulator extends CameraManipulator
{
    public void setHomePosition(Vector3f eye, Vector3f center, Vector3f up);
    public void getHomePosition(Vector3f eye, Vector3f center, Vector3f up);
}

class VRPNManipulator extends CameraManipulator
{
    public enum DeviceType {SPACE_MOUSE, WIIMOTE}
    public VRPNManipulator(DeviceType type, String hostName);
}

class Pointer
{
}

class WiimotePointer extends Pointer
{
    public WiimotePointer(String hostName);
}
