Federated learning process
Folder system description:
Federated_learning_case : This folder will house the server script for the federated learning process
FL_client : This folder will house the client script for the federated learning process

The frame work for the federated learning process can be seen in image FL_framework_withDRL.png. The current framework of this folder DO NOT include the DRL component in the server script
FL_1.png shows the results of the FL script.(Image description is as follows)
1) The time taken on the left is the training time for the RPI3 on 35% of the whole covid 19 set (Both dataset are mutually exclusive)
2) The time taken on the right is the training time for the RPI4 on 35% of the whole covid 19 set (Both dataset are mutually exclusive)
3) The cmd from left to right shows the accuracy of the averaged global model.
- The first cmd shows the accuracy of a randomly initialized global model with the model in RPI 3
- The second cmd shows the accuracy of a randomly initialized global model with the model in RPI4
- The third cmd shows the accuracy of a randomly initialized global model with both the model of RPI 3 and RPI4

Procedure to reproduce result
Server side:
1) Change host IP address and port number on the orchestrator.py script (FL_case\Federated_learning_case\orchestrator)
2) Run the orchestrator script
- this will initiate the server script in a run till cancel loop

Client side:
1) Change the URI address to the server IP address in the client_code.py (FL_case\FL_client)
2) Run the client_code.py

Additional notes:
1) Image dataset is not provided, preprocessing such as resizing is done in the pytorch_mobilenetv2_model.py script
- Add the training images to the train_covid_folder
- Add the testing images to the test_covid_folder
2) If the Client cannot connect to the server (Noticeable especially when the client script does not produce response) try the following:
- Turn off the firewall on the server host machine
- Make sure the port of the server host machine is actively listening (do a ping)