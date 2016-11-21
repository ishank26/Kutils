from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC 


def scale_features(data):
    extract_features = theano.function([model.layers[0].input],model.layers[32].output,allow_input_downcast=True)
    features = extract_features(data)
    scale=MinMaxScaler()
    scale_feat= scale.fit_transform(features)
    return scale_feat

print "scaling train feats"
train_feats=scale_features(X_train)
print " scaling test feats"
test_feats=scale_features(X_test)    

svc=SVC(cache_size=60000)

print "Fitting svm"
svc.fit(train_feats,y_train)

print"Making predictions"
pred=svc.predict(test_feats)

def predictions(pred,y_test):
    positive=[]
    for i in range(len(y_test)):
        if pred[i]==y_test[i]:
            positive.append(int(i))
    return positive, len(positive)/float(len(y_test)) 

positives, accuracy = predictions(pred, y_test)

print "accuarcy: ", accuracy
print "positives: ",positives
