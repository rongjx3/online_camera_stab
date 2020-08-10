package me.zhehua.gryostable.widget;

import android.app.Activity;
import android.content.Context;
import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLContext;
import android.opengl.GLES30;
import android.opengl.GLSurfaceView;
import android.os.SystemClock;
import android.util.Log;

import java.util.Arrays;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import me.zhehua.gryostable.filter.CameraFilter;
import me.zhehua.gryostable.filter.ScreenFilter;
import me.zhehua.gryostable.record.AvcRecorder;
import me.zhehua.gryostable.util.Camera2Helper;
import me.zhehua.gryostable.util.OnRecordListener;
import me.zhehua.gryostable.util.OpenGlUtils;

public class GlRenderWrapper implements GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener,
        Camera2Helper.OnPreviewListener, Camera2Helper.OnPreviewSizeListener{

    private final String TAG = "GlRenderWrapper";
    private final GlRenderView glRenderView;
    public Camera2Helper camera2Helper;
    private int[] mTextures;
    private SurfaceTexture mSurfaceTexture;
    public ScreenFilter screenFilter;
    public CameraFilter cameraFilter;
    private int mPreviewWdith;
    private int mPreviewHeight;
    private int screenSurfaceWid;
    private int screenSurfaceHeight;
    private int screenX;
    private int screenY;
    private AvcRecorder avcRecorder;
    private OnRecordListener onRecordListener;
    private float[] mtx = new float[16];
    public boolean isready = false;

    public long timeStamp = 0;

    private long renderTimeStamp = 0;
    private long lastRenderTimeStamp = 0;

    public GlRenderWrapper(GlRenderView glRenderView){
        this.glRenderView = glRenderView;
        Context context = glRenderView.getContext();

    }

    @Override
    public void onSurfaceCreated(GL10 gl10, EGLConfig eglConfig) {
        camera2Helper = new Camera2Helper((Activity)glRenderView.getContext(), glRenderView);
        mTextures = new int[2];
//        GLES30.glGenTextures(mTextures.length, mTextures, 0);
        OpenGlUtils.glGenTextures(mTextures);
        mSurfaceTexture = new SurfaceTexture(mTextures[0]);
//        mSurfaceTexture.setOnFrameAvailableListener(this);

        //使用fbo 将samplerExternalOES 输入到sampler2D中
        cameraFilter = new CameraFilter(glRenderView.getContext());
        //负责将图像绘制到屏幕上
        screenFilter = new ScreenFilter(glRenderView.getContext());

    }

    @Override
    public void onSurfaceChanged(GL10 gl10, int width, int height) {
        camera2Helper.setPreviewSizeListener(this);
        camera2Helper.setOnPreviewListener(this);
        camera2Helper.openCamera(width, height, mSurfaceTexture);
        float scaleX = (float) mPreviewHeight /(float) width;
        float scaleY = (float) mPreviewWdith / (float) height;
        float max = Math.max(scaleX, scaleY);
        screenSurfaceWid = (int) (mPreviewHeight / max);
        screenSurfaceHeight = (int) (mPreviewWdith / max);
        screenX = width - (int) (mPreviewHeight / max);
        screenY = height - (int) (mPreviewWdith / max);

        cameraFilter.prepare(screenSurfaceWid, screenSurfaceHeight, screenX, screenY);
        screenFilter.prepare(screenSurfaceWid, screenSurfaceHeight, screenX, screenY);

        EGLContext eglContext = EGL14.eglGetCurrentContext();

        avcRecorder = new AvcRecorder(glRenderView.getContext(), 1080, 1920, eglContext);
        avcRecorder.setOnRecordListener(onRecordListener);

        isready = true;
    }

    @Override
    public void onDrawFrame(GL10 gl10) {
        int[] textureId;
//        GLES30.glClearColor(0 ,0, 1, 0);
//        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
//        mSurfaceTexture.updateTexImage();
//
        renderTimeStamp = SystemClock.elapsedRealtimeNanos();
        Log.d(TAG, "fps render1111: "+ 1/((float)(renderTimeStamp - lastRenderTimeStamp)/1000/1000/1000));
        lastRenderTimeStamp = renderTimeStamp;

        mSurfaceTexture.getTransformMatrix(mtx);
//        Log.d(TAG, "onDrawFrame: "+Arrays.toString(mtx));
        //cameraFiler需要一个矩阵，是Surface和我们手机屏幕的一个坐标之间的关系

//        cameraFilter.setMatrix(mtx);
        int[] id = screenFilter.onDrawFrame(mTextures);
//        textureId = cameraFilter.onDrawFrame(mTextures);

        Log.d(TAG, "onDrawFrame22222: " + Arrays.toString(id));

//        给录制的filter传数据
        if(avcRecorder.eglConfigBase != null ){
            if(avcRecorder.eglConfigBase.recordFilter != null){
                avcRecorder.eglConfigBase.recordFilter.transformMatrix = screenFilter.transformMatrix;
                avcRecorder.eglConfigBase.recordFilter.rsMat = screenFilter.rsMat;
            }

        }
//        if(avcRecorder.eglConfigBase != null){
//            if(avcRecorder.eglConfigBase.recordFilter != null){
//                avcRecorder.eglConfigBase.recordFilter.setIsOpenRollingShutter(false);
//            }
//        }

        boolean flag = avcRecorder.encodeFrame(id, timeStamp);
        if(flag){
            timeStamp += 33000000;
        }




    }

    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
//        glRenderView.requestRender();
    }

    public void onSurfaceDestory() {
        if (camera2Helper != null) {
            camera2Helper.closeCamera();
            camera2Helper.setPreviewSizeListener(null);
        }


        if (cameraFilter != null)
            cameraFilter.release();
        if (screenFilter != null)
            screenFilter.release();
    }

    @Override
    public void onSize(int width, int height) {
        mPreviewWdith = width;
        mPreviewHeight = height;

    }

    public void startRecord(float speed, String path) {
        avcRecorder.start(speed, path);
    }

    public void stopRecord() {
        avcRecorder.stop();
    }

    @Override
    public void onPreviewFrame(byte[] data, int len) {

    }

    public void setOnRecordListener(OnRecordListener onRecordListener) {
        this.onRecordListener = onRecordListener;

    }
    public void getTimeStamp(long timestamp){
        this.timeStamp = timestamp;
    }


}
