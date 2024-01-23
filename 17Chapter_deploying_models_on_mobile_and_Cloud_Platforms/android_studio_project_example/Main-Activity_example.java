public class MainActivity extends AppCompatActivity
{
	private ImageView imgCapture; 
	private static final int Image_Capture_Code = 1; 
	...
	@Override 
	protected void onCreate(bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState); 

		initClassifier(getAssets()); 

		setContentView(R.layout.activity_main); 
		imgCapture = findViewById(R.id.captureImage); 
		Botton btnCapture = findViewById(R.id.btnTakePicture); 
		btnCapture.setOnClickListener(new View.OnClickListener()
				{
					@Override
					public void onClick(View v)
					{
						Intent cInt = new Intent(MediaStore.ACTION_IMAGE_CAPTURE); 
						startActivityForResult(cInt, Image_Capture_Code); 
					}
				}); 
	}
	...
}

/*we identified the image that was captured with our previously defined "Image_Capture_Code" code and passed it into the "startActivityForResult()" method with the intent class object "cInt".*/

/*Next, is the "onActivityResult" which is launched when "startActivityForResult()" method captures a image on software and passes it to "startActivityForResult()"*/
/*"OnActivityResult()" method processes the application results.*/
@overrid
protected void onActivityResult(int requestCode, int resultCode, Intent data)
{
	if (requestCode == Image_Capture_Code)
	{
		if (resultCode == RESULT_OK)
		{
			Bitmap bp = (Bitmap) Objects.requireNonNull(data.getExteras()).get("data"); 
			if (bp != null)
			{
				Bipmap argb_bp = bp.copy(Bitmap.Config.ARG_8888, true); 
				if (argb_bp != null)
				{
					float ratio_w = (float) bp.getWidth() / (float) bp.getHeight(); 
					float ratio_h = (float) bp.getHeight() / (float) bp.getWidth(); 

					int width = 224; 
					int height = 224; 

					int new_width = Math.max((int) (height * ratio_w), width); 
					int new_height = Math.max(height, (int) (width * ratio_h)); 

					Bitmap resized_bitmap = Bitmap.createScaledBitmap(argb_bp, new_width, new_height, false); 
					Bitmap cropped_bitmap = Bitmap.createBitmap(resized_bitmap, 0, 0, width, height); 

					int[] pixels = new int[width * height];
					cropped_bitmap.getPixels(pixels, 0, width, 0, 0, width, height); 
					String class_name = classifyBitmap(pixels, width, height); 
					imgCapture.setImageBitmap(cropped_bitmap); 

					TextView class_view = findViewById(R.id.textViewClass); 
					class_view.setText(class_name); 
				}
			}
		} else if (resultCode == RESULT_CANCELED)
		{
			Toast.makeText(this, "Cancelled", Toast.LENGTH_LONG).show(); 
		}
	}
}

/*The "Bitmap getPixels()" method was used to get raw color values from the "bitmap". */

/*There are two methods. "classifyBitmap" and "initClassifier", which are JNI calls to the native library functions that are implemented with C++. 
 * TO connect to native library with the Java code, we use the Java Native Interface(JNI). 
 * This is a standard mechanism that's used for calling C/C++ functions from Java. 
 * First, we have to load the native library with the "system.LoadLibrary" call. 
 * Then, we have to define the methods that are implemented in the native library by declaring them as "public native".*/
public class MainActivity extends AppCompatActivity
{
	...
		static
		{
			System.loadLibrary("native-lib"); 
		}

	public native String classifyBitmap(int[] pixels, int width, int height); 
	public native void initClassifier(AssetManger assetManager); 
	...
}

/*Notice that we called the "initClassifier()" method in the "onCreate()" method and passed it into the "AssetManager" object, which was returned by the "getAssets Activity()" method.*/
