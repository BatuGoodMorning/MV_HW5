#include "hw5.h"

int K = 100;
//define Term Criteria
TermCriteria tc(0,100,0.001);
//retries number
int retries=1;
//necessary flags
int flags=KMEANS_PP_CENTERS;

Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);  
BOWImgDescriptorExtractor bowDE( extractor, matcher );
//Ptr<FeatureDetector> detector(new SiftFeatureDetector());
Ptr<SIFT> detector;

BOWKMeansTrainer bowTrainer(K,tc,retries,flags);
String folderNames[5] = {"airCond", "Car", "centHeat", "tree", "Umbrella"};
String imgPath;
int labels[5] = {1, 2, 3, 4, 5};


#define DICTIONARY_BUILD 0 // set DICTIONARY_BUILD to 1 for Step 1. 0 for step 2 

int main()

{

    //#if DICTIONARY_BUILD == 1
        detector = SIFT::create();
        cout << "Starting to get SIFT descriptors for each image!!" << endl;

    for (int i = 0; i<5; i++) 
    {
        for (int j =1; j<31; j++)
        {
            Mat input;  
            vector<KeyPoint> keypoints;
            Mat descriptors;
            imgPath = folderNames[i] + "/Train/" +to_string(j) +".jpg";
            input = imread(imgPath, IMREAD_GRAYSCALE);
            detector->detectAndCompute(input, noArray(), keypoints, descriptors);
            if (!descriptors.empty()) bowTrainer.add(descriptors);
        }
    }

        cout << "Dictionary is obtained !!" << endl;
        cout << endl;

        // Create the vocabulary with KMeans.
        cout << "Clustering features, this might take a while... " << endl;
        Mat vocabulary;
        vocabulary = bowTrainer.cluster(); 
        bowDE.setVocabulary(vocabulary); 
        //store the vocabulary
        FileStorage fs("dictionary.xml", FileStorage::WRITE);
        fs << "vocabulary" << vocabulary;
        fs.release();

        cout << "Clustering complete! Vocabulary is written to the dictionary file!" << endl;
    
    
    //cluster the feature vectors

    
    //Step 2 - Obtain the BoF descriptor for given image/video frame. 

    //prepare BOW descriptor extractor from the dictionary
    //#else    
        Mat dictionary, train, label;
        /*
        FileStorage fs("dictionary.xml", FileStorage::READ);
        fs["vocabulary"] >> dictionary;
        fs.release();
        bowDE.setVocabulary(dictionary);
        */
        //detector = SIFT::create();

        

        //Set the dictionary with the vocabulary we created in the first step
        for (int i=0; i<5; i++)
        {
            for (int j=1; j<31; j++)
            {
                Mat input;  
                vector<KeyPoint> keypoints;
                Mat descriptors, bowDescriptors;
                
                imgPath = folderNames[i] + "/Train/" +to_string(j) +".jpg";
                input = imread(imgPath, IMREAD_GRAYSCALE);
                namedWindow("Input", WINDOW_AUTOSIZE);
                imshow("Input", input);
                waitKey(0);
                destroyAllWindows();
                detector->detectAndCompute(input, noArray(), keypoints, descriptors);
                cout << "OKK" << endl;
                bowDE.compute(input, keypoints, bowDescriptors);
                cout << "SOKK" << endl;
                if (!bowDescriptors.empty())
                {
                    train.push_back(bowDescriptors);  // update training data
                    label.push_back(labels[i]);       // update response data
                }
                cout<<"LLOKK"<<endl;
            }
        }
        
        cout << "Training... ";
        // setup svm as per tutorial values
        String svmSave;
        svmSave = "svmModel.xml";
        Ptr<SVM> svm = SVM::create();
        svm->setType(ml::SVM::C_SVC);
        svm->setKernel(ml::SVM::RBF);
        svm->setGamma(8);
        svm->setDegree(10);
        svm->setCoef0(1);
        svm->setC(10);
        svm->setNu(0.5);
        svm->setP(0.1);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
        svm->train(train, ROW_SAMPLE, label);
        svm->save(svmSave);

        cout << "Training complete!" << endl;
        cout << "SVM saved at " << svmSave << endl;

    //#endif

}