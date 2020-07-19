


































































	# if our correlation object tracker is None we first need to
	# apply an object detector to seed the tracker with something
	# to actually track
	if tracker is None:
		# grab the frame dimensions and convert the frame to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
		# pass the blob through the network and obtain the detections
		# and predictions
		net.setInput(blob)
		detections = net.forward()


		# ensure at least one detection is made
		if len(detections) > 0:
			# find the index of the detection with the largest
			# probability -- out of convenience we are only going
			# to track the first object we find with the largest
			# probability; future examples will demonstrate how to
			# detect and extract *specific* objects
			i = np.argmax(detections[0, 0, :, 2])

			# grab the probability associated with the object along
			# with its class label
			conf = detections[0, 0, i, 2]
			label = CLASSES[int(detections[0, 0, i, 1])]

			# filter out weak detections by requiring a minimum
			# confidence
			if conf > args["confidence"] and label == args["label"]:
				# compute the (x, y)-coordinates of the bounding box
				# for the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				# draw the bounding box and text for the object
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(frame, label, (startX, startY - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# otherwise, we've already performed detection so let's track
	# the object
	else:
		# update the tracker and grab the position of the tracked
		# object
		tracker.update(rgb)
		pos = tracker.get_position()

		# unpack the position object
		startX = int(pos.left())
		startY = int(pos.top())
		endX = int(pos.right())
		endY = int(pos.bottom())

		# draw the bounding box from the correlation object tracker
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		cv2.putText(frame, label, (startX, startY - 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break