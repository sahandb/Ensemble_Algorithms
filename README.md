# Ensemble Algorithms (AdaBoost.M1 and Bagging)
Implementation of two ensemble algorithms, Bagging, and AdaBoost.M1 for imbalanced data.

# Method I: Adaboost with UnderSampling
The motivation of this method is to keep the high efficiency of under-sampling but reduce the possibility of ignoring potentially useful information contained in the majority class examples.
This method randomly generates multiple sample sets {π1, π2, . . . , ππ } from the majority class π.
The size of each sample set is equal to the size of the minority class, i.e., |ππ | = |π|.
In each iteration, the union of pairs ππ and π is used to train the AdaBoost ensemble.
The final ensemble is formed by combining all the base learners in all the AdaBoost ensembles.
The algorithm is shown in Figure below.

![image](https://user-images.githubusercontent.com/24508376/219408652-b8d6d1ea-9a15-4bc1-b82a-a7a33f848827.png)

This algorithm generates T balanced sub-problems. The output of the ππ‘β sub-problem is AdaBoost classifier π»π , an ensemble with π π weak classifiers {βπ,π }. Finally, instead of counting votes from {π»π }π=1,β¦,π , we collect all the βπ,π (π = 1, 2, . . ., π; π = 1, 2, . . . , π π ) and form an ensemble classifier out of them.

Initialize weights equal to 1/n and every time use weak classifier (decision stump) for train the weight with check the predicted and real label and if it missed the label the weight must update and if its correct the value of alpha lose and else it gain and then the training weight giving more weight if its incorrect and lose weight if it correct and weights update like this till training finishd and then predict the efficiency of algorithm and then calculate the metrics for 11 and 31 and 51 and 101.

The output of the above algorithm is a single ensemble, but it looks like an βensemble of ensemblesβ.

Note 1:
The number of iterations π, and the number of rounds π π selected from the sets {10, 15} and {11, 31, 51, 101}, respectively. The classification results of all these parameters  provided in our report.

# Method II: Bagging with UnderSampling

For each iteration in Bagging, draw a bootstrap sample from the minority class. Randomly draw the same number of samples, with replacement, from the majority class. Repeat this step for the π number of times. Aggregate the predictions of base learners and make the final prediction.

Note 2: In the Bagging Algorithm, the number of learners or iterations π should be selected from the set {11, 31, 51, 101}, and the results reported accordingly.

We use the implemented decision tree as the base learner of our ensemble models. max_depth parameter of the base learners in the AdaBoost.M1 algorithm should be tuned experimentally so that the decision tree performs a little better than a random classifier. For the Bagging algorithm, we use the default parameters of the base learner decision tree.

every time use classifier for train the model and get predict every time for test then get vote from the list of predictions and then calculate the accuracy for all and other metrics for the minority class and run the algorithm and then calculate the metrics for 11 and 31 and 51 and 101.


We splited the data set into train and test parts. Used 70% of the data for the training phase and the remained 30% for the testing phase. Run our codes for 10 individual runs and reported the mean and standard deviation of 10 runs for each performance metric.

Since the data set is imbalanced, you should use Precision, Recall, F-measure, and AUC measures to evaluate the performance of these algorithms. It is worth mentioning that these measures are one-class measures; in other words, you should compute these metrics just for the minority class. Moreover, G-mean should be reported.
___
___

π»πππ π·πππππππ πΉπππ (π¨ππ+)=π»π·/(π»π·+π­π΅)

___

π»πππ π΅πππππππ πΉπππ (π¨ππβ)=π»π΅/(π»π΅+π­π·)

___

π?βππππ=βπ¨ππ+Γπ¨ππβ

___

Compared the performance of our ensemble methods with the following classifiers: NaΓ―ve Bayes, One-nearest-neighbor (OneNN), linear SVM, and kernel SVM (use RBF kernel). For each classifier, used the balanced version of the data by the oversampling and undersampling techniques.
Under-sampling deletes majority examples from the dataset so that the number of examples between different classes becomes balanced.
Over-sampling is to add minority examples to the dataset to achieve a balance, in which the existing minority examples are replicated, or artificial minority examples are generated.

# Some important questions:
Why  should  we  set max_depth  parameter  in  AdaBoost.M1  so  that  the  base  classifiers become a little better than random guess?
Because we need to only have better performance than random guess and it happened when we have weak classifier and only have 1node with 2 leaves (decision stump) for that and we have more than one weak learner and update them every step.

What do we mean by stable, unstable, and weak classifier?
A stable learning algorithm is one for which the prediction does not change much when the training data is modified slightly.
And if learning algorithm is stable, changing input slightly won't put effect on the hypothesis. In unstable, it would.
classifiers that tend to overfit (high variance) are unstable.
A weak classifier (decision stump) is preparedΒ on the training data using the weighted samples and with the rate of error get update the weighted samples.

What kind of classifiers should be used in Bagging? How about AdaBoost.M1?Why?
In bagging we use strong classifier not weak and then get voting at the end and in Adaboost we use weak classifier because we just need better performance than random guess.
