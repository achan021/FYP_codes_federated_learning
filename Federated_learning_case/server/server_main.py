from model_scripts import keras_lenet_model, pytorch_lenet_model,pytorch_inception_model,pytorch_mobilenetv2_model
import os

def main(library_sel,mode):
    if library_sel == 0:

        #run pytorch loading and model creation
        # load input dataset
        classes, trainset, testset = pytorch_lenet_model.load_dataset()

        if mode == 0:
            #train the model
            net = pytorch_lenet_model.get_net()
            net = pytorch_lenet_model.train_model(net,trainset)
            #save the model
            pytorch_lenet_model.save_model(net)

        elif mode == 1:
            #load the model
            net = pytorch_lenet_model.get_net()
            net = pytorch_lenet_model.load_model(net)
            #test the model
            pytorch_lenet_model.test_model(net,testset,classes)

        else:
            print("wrong mode selection!!")

    elif library_sel == 1:
        #run keras loading and model creation
        # load input dataset
        classes, trainX, trainY, testX, testY = keras_lenet_model.dataloader_preprocess_keras()
        # load the model
        net = keras_lenet_model.leNet_keras(len(classes),trainX.shape[1:])

        if mode == 0:
            #train the model
            net = keras_lenet_model.train_model_keras(net,trainX,trainY)
            #save the model
            keras_lenet_model.save_model(net)
        elif mode == 1:
            #load the model
            net = keras_lenet_model.load_model()
            #test the model
            keras_lenet_model.test_model(testX,testY,net)

        else:
            print("wrong mode selection!!")

    elif library_sel == 2: #select pytorch inceptionv3 model
        model = pytorch_inception_model.get_net()
        train_loader , test_loader = pytorch_inception_model.load_dataset()
        if mode == 0:
            #train the model
            pytorch_inception_model.train_model(model,train_loader)
            pytorch_inception_model.save_model(model)
        elif mode == 1:
            pytorch_inception_model.load_model(model)
            loss, accuracy = pytorch_inception_model.evaluate(model,test_loader)
            print('evaluation loss : {} ---- evaluation accuracy : {}'.format(loss,accuracy))
        else:
            print("wrong mode selection!!")

    elif library_sel == 3:#select pytorch mobilenetv2 model
        model = pytorch_mobilenetv2_model.get_net()
        train_loader, test_loader = pytorch_mobilenetv2_model.load_dataset()
        if mode == 0:
            # train the model
            pytorch_mobilenetv2_model.train_model(model, train_loader)
            pytorch_mobilenetv2_model.save_model(model)
        elif mode == 1:
            pytorch_mobilenetv2_model.load_model(model)
            loss, accuracy = pytorch_mobilenetv2_model.evaluate(model, test_loader)
            print('evaluation loss : {} ---- evaluation accuracy : {}'.format(loss, accuracy))
        else:
            print("wrong mode selection!!")

    else:
        print('Error selection')

#interface between pytorch and keras model
library_sel = int(input("0 : pytorch , 1 : keras , 2 : inception(pytorch) , 3 : mobilenetv2(pytorch)"))
mode = int(input("0 : train new model, 1 : test available model"))
main(library_sel,mode)