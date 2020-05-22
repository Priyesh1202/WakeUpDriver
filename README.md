# WakeUpDriver

This project was to capture a driver's live web cam feed and after processing every frame output whether or not the driver is on the verge of drifting away.

I focused on two factors: 
1. Eyes: The vertical distance between the facial landmark points for an eye will help determine whether his eyes are wide open or not.
2. Yawns: Yawns either of boredom or tiredness. Either way it is fatal as it leaves the driver distracted for a moment.The vertical distance between the upper and lower lips helps us determine whether its a yawn or not.

Whenever the program observes closing eyes or a yawn , it will start to play an alarm.

This project is based on dlibs facial landmarks model. On detecting a face it will map the key points which will be used by us for the above 2 points.

To run :  python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
