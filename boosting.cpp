


using namespace std;
#include <iostream>
#include <set>
#include <utility>
#include <vector>
#include <algorithm>
#include <map>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <bits/stdc++.h>
#define ff first
#define ss second
#define pb push_back
int ct = 0,bt = 0,edges = 0;
char tempstore[300];
vector<vector<string> >trainingdata,testingdata;
vector<set<string> >dataset(15),testdataset(15);  ///contain all possible data of each attribute
int numberOfAttributes = 15;
double boost[400][41000] = {1.0};
//int data_weight[41000] = {1};
double classifier_weight[400];
int hit[400][41000];
double wrong_hit[400];
struct attribute;
struct internalnode;

struct attribute{
    int attribute_num; // Attribute number
    int answer;
    bool leaf;  //to check whether its a leaf or not
    vector<internalnode *>next;   //pointers to all internal nodes
};
vector<attribute *>treenodepointers;

struct internalnode{
    int node_num;
    attribute *nxt;
};
int usedboost[400];
vector<string>threshold[15];
double entropy(vector<vector<string> > data){
    int pos, neg, total;
    pos = neg = total = 0;
        int len = data.size()-1;

    for(int i=0 ; i<data.size() ; i++){
        vector<string>vec = data[i];
        if(vec[14] == ">50K")
            pos++;
        else 
            neg++;
    }
  
    total = pos + neg;
     
    if(pos == 0 || neg == 0)
        return 0.0;

    int x = pos, y = neg;
    
    double p1, p2;
    p1 = (1.0 * pos) / total, p2 = (1.0 * neg) / total;
    
    return -1.0 * (p1 * log2(p1) + p2 * log2(p2));
}

double entropy(int greater, int lesser){
    int total = greater + lesser;
    
   if(greater == 0 || lesser == 0)
        return 0.0;
    
    double p1, p2;
    p1 = (1.0 * greater) / total, p2 = (1.0 * lesser) / total;
    
    return -1.0 * (p1 * log2(p1) + p2 * log2(p2));
}

int inf_gain(vector<vector<string> > data, set<int> used,set<int>randomnumbers,set<int>randomrows){
    bool flag = false;
    int index = -1;
    double val = 0.0, entrpy = 0.0, tempEntropy;

    entrpy = entropy(data);
 
    if(entrpy == 0)
        return index;
    
  for(int attribute = 0 ; attribute < numberOfAttributes - 1; attribute++){
        if(randomnumbers.find(attribute)!=randomnumbers.end() && used.find(attribute) == used.end()){
            int len = dataset[attribute].size();
          
            int cnt[len][2], total = 0;
            
            for(int j=0 ; j<len ; j++)
                cnt[j][0] = cnt[j][1] = 0;

            int j = 0;
            int k = 0;
            for(set<string>::iterator it = dataset[attribute].begin();it!=dataset[attribute].end();it++){
                //for(int k=0 ; k<data.size() ; k++){
                  for(set<int>::iterator bt = randomrows.begin();bt!=randomrows.end();bt++){k = *bt;
                    if(data[k][attribute] == *it){
                        total++;
                        if(data[k][14] == ">50K"){
                            cnt[j][1]++;
                        } else {
                            cnt[j][0]++;
                        }
                    }
                }
                j++;
            }

            if(total == 0)
                continue;
            
            tempEntropy = entrpy;
           
           
            for(int j=0 ; j<len ; j++){
                tempEntropy -= ((cnt[j][0] * 1.0 + cnt[j][1]) / total) * entropy(cnt[j][0], cnt[j][1]);
            }
           
            
           if(flag){
                if(tempEntropy > val){
                    val = tempEntropy;
                    index = attribute;
                }
           } else {
               val = tempEntropy;
                flag = true;
               index = attribute;
            }
        }
    }
    
    return index;
}


void ID3(attribute **root,vector<vector<string> >data,set<int>used){
    set<int>randomnumbers,randomrows;
    int num = rand()%14;
   
        
        randomnumbers.insert(num);
        randomnumbers.insert((num+2)%14);
        randomnumbers.insert((num+4)%14);
        randomnumbers.insert((num+8)%14);
        randomnumbers.insert((num+11)%14);
        randomnumbers.insert((num+15)%14);
 
        int row = rand()%data.size();
        for(int i = 0; i<data.size()/6;i++){
            row+=6;
            row%=data.size();
            randomrows.insert(row);
        }
  
    int att_used = inf_gain(data,used,randomnumbers,randomrows);

    *root = new attribute();
  
    (*root)->leaf =false;

    (*root)->attribute_num = att_used;
     int output[2] = {0};
     if(att_used == -1){
        (*root)->leaf = true;
        
    }
    for(int i = 0; i<data.size();i++){
       if(data[i][14] == ">50K")output[1]++;
       else
        output[0]++;

    }
    
if(output[1]>=output[0])
    (*root)->answer = 1;
else
    (*root)->answer = 0;



      if(att_used == -1)
        return;

    *root = new attribute();
    (*root)->leaf =false;

    (*root)->attribute_num = att_used;
        
    int len = dataset[att_used].size();

    vector<vector<vector<string> > >tempdataused(len);
    int j = 0;

    for(set<string>::iterator it = dataset[att_used].begin();it!=dataset[att_used].end();it++){
            string str = *it;
        
            for(int i = 0; i<data.size();i++){
                if(data[i][att_used] == str)
                    tempdataused[j].pb(data[i]);
            }
            j++;
    }

    used.insert(att_used);

    for(int i=0 ; i<len ; i++){
       
        internalnode *in = (internalnode *)malloc(sizeof(internalnode));
        in->node_num = i;      
        in->nxt = NULL;
        (*root)->next.pb(in);        
    }

    for(int i=0 ; i<len ; i++){
        if(tempdataused[i].size()){
          ID3(&((*root)->next[i]->nxt),tempdataused[i], used);
        }
   }
}


void updateweight(int index,double alpha){    //updating weight of data
    int t = 0;
    for(int i = 0; i<trainingdata.size();i++){
        if(hit[index][i] == 0)boost[index][i] = 1.0* exp(1.0*alpha)*boost[index][i];
        else
            boost[index][i] =  1.0*(boost[index][i])/exp(1.0*alpha);
        
    }
    
}


double calculate_alpha(double wrong,double total){
    double em = 1.0*(wrong)/(total);
    double temp = 1.0*(1-em)/em;
    return log(temp);
}

void assign_weight(){                  //assigning weight to the classifier

    for(int t = 0; t<7;t++){

        double wrongweightsum[400] = {0.0},totalweightsum[400] = {0.0};
        for(int i = 0; i<treenodepointers.size(); i++)
            wrongweightsum[i] = totalweightsum[i] = 0.0;
      
            for(int tree = 0; tree<treenodepointers.size(); tree++){
                for(int data = 0; data<trainingdata.size();data++){  
                    totalweightsum[tree]+=boost[tree][data];
                        if(hit[tree][data] == 0)
                            wrongweightsum[tree]+=boost[tree][data];
            }
        }
      

        int index = -1,min_weight  = 1;
         for(int sum = 0; sum<treenodepointers.size();sum++)
            if(!usedboost[sum]){min_weight = wrongweightsum[sum];break;}
                
        

        for(int sum = 0; sum<treenodepointers.size();sum++){
            if(!usedboost[sum] && wrongweightsum[sum]<=min_weight){
                index = sum;
                min_weight = wrongweightsum[sum];
            }
        }
 
        if(index==-1 || usedboost[index]==1){
            continue;
        }
       double alpha = calculate_alpha(wrongweightsum[index],totalweightsum[index]);
     
      
        updateweight(index,alpha);
        usedboost[index] = 1;
        classifier_weight[index] = alpha;
    }

}

int search(attribute **root,vector<string>test){   //Testing
    if(*root == NULL)
        return 0;
    if((*root)->leaf == true){

        int temp = (*root)->answer;
        return temp;
    }

    int index = (*root)->attribute_num;

    string str = test[index];
   
    int j = 0;
  
    for(set<string>::iterator i = dataset[index].begin(); i!=dataset[index].end();i++){
            if(*i == str){ break;}
            j++;
    }
 
    return search(&((*root)->next[j])->nxt,test);
    
}




int main(int argc, char const *argv[])
{
     int flag = 0;
    FILE *inputfile = fopen("adult.txt","r+");
    for(int i = 0; i<331;i++)
        for(int j = 0; j<40000;j++)
            boost[i][j] = 1.0;

    for(int i = 0; i<301;i++)
        classifier_weight[i] = 1.0;

    while(fscanf(inputfile, "%[^\n]s",tempstore)!=EOF){
        int len = strlen(tempstore),j = 0;
        vector<string>strtemp;
      
        while(tempstore[j] == ' ')j++;
        int p = 0;
        string temp = "";
        for(;j<len;j++){
            if(tempstore[j] ==' ' || tempstore[j] ==','){
                if(temp!=""){
                strtemp.pb(temp);
                threshold[p].pb(temp);
                p++;
                
                }
                temp = "";
                continue;
            }
            temp.pb(tempstore[j]);
            
        }
        strtemp.pb(temp);
        trainingdata.pb(strtemp);

        int lastinserted = trainingdata.size()-1;
      
        for(int i = 0; i<trainingdata[lastinserted].size();i++){
         string str = trainingdata[lastinserted][i];
         if(str!="?")
          dataset[i].insert(str);    
        }
       fgetc(inputfile);

    }

     for(int i = 0; i<15; i++){
                 if( i == 0 || i == 2 || i == 4 || i == 10 || i == 11 || i == 12)continue;
                int len = dataset[i].size();
                int ct[len];
            for(set<string>::iterator it = dataset[i].begin();it!=dataset[i].end();it++){
                
                int pos = distance(dataset[i].begin(),it);//(it - dataset[i].begin());
         
                    for(int t = 0; t<len ;t++)ct[t] = 0;
                        for(int j = 0; j<trainingdata.size();j++){
                            if(trainingdata[j][i] == *it)
                            ct[pos]++;
                        }
                }

                        int index = -1,maxvalue = -1;
                            for(int t = 0; t<len;t++){
                                    if(ct[t]>maxvalue){
                                        maxvalue = ct[t];
                                        index = t;
                                    }
                            }

                    string copy_string = "";
                    int val = 0;
                    for(set<string>::iterator it = dataset[i].begin();it!=dataset[i].end();it++){
                            if(val == index){
                                copy_string = *it;
                                break;
                            }
                            val++;
                    }
                  

                   for(int t = 0 ; t<trainingdata.size();t++)

                        if(trainingdata[t].size() == 15 && trainingdata[t][i] == "?" ){
                         
                            trainingdata[t][i] =""; //copy_string;
                            trainingdata[t][i] +=copy_string;  
                        }
        
    }
        
       int arr[15] = {0};
    for(int i = 0; i<trainingdata.size();i++){
        for(int t = 0; t<15; t++)
            if( t == 0 || t == 2 || t == 4 || t == 10 || t == 11 || t == 12){
                string str = trainingdata[i][t];
                if(str == "?")continue;
                char tempstring[100];
                for(int pp = 0; pp <str.size();pp++)
                    tempstring[pp] = str[pp];
                    int x = atoi(tempstring);
                    arr[t]+=x;
            }
    }
    int trainingdatasize = trainingdata.size();

    for(int i = 0; i<trainingdatasize;i++){
        for(int t = 0; t<trainingdata[i].size();i++)
          if( t == 0 || t == 2 || t == 4 || t == 10 || t == 11 || t == 12){
               
            if(trainingdata[i][t] == "?"){ 
                int temp = (arr[t]/trainingdatasize);
                stringstream ss;
                ss<<temp;
                trainingdata[i][t] =ss.str();
          }
    }
    }

    
   for(int i = 0; i<15; i++)
        sort(threshold[i].begin(),threshold[i].end());


    for(int i = 0; i<15; i++){

        if( i ==0 || i == 2 || i == 4 || i == 10 || i == 11|| i == 12){
            dataset[i].clear();
            dataset[i].insert("0");
            dataset[i].insert("1");
        }
        
    }   
 
    dataset[0].insert("2");
    dataset[0].insert("3");

int a,b,c,d,e,f = 0,g = 0;
            for(int j = 0; j<trainingdata.size();j++){
                for(int t = 0; t<trainingdata[j].size();t++){

                        if(t == 0){
                            if(trainingdata[j][t]<="30")trainingdata[j][t] = "0";
                            else if(trainingdata[j][t]<="45")trainingdata[j][t] = "1";
                            else if(trainingdata[j][t]<="60")trainingdata[j][t] = "2";
                            else if(trainingdata[j][t]<="90")trainingdata[j][t] = "3";
                        }
                        if(t == 2){
                            if(trainingdata[j][t]<="60000")trainingdata[j][t] = "0";
                            else
                                trainingdata[j][t] = "1";
                        }
                         if(t == 4){
                            if(trainingdata[j][t]<="8")trainingdata[j][t] = "0";
                            else
                                trainingdata[j][t] = "1";
                        }

                         if(t == 10){
                            if(trainingdata[j][t]<="10000")trainingdata[j][t] = "0";
                            else
                                trainingdata[j][t] = "1";
                        }
                         if(t == 11){
                            if(trainingdata[j][t]<="2000")trainingdata[j][t] = "0";
                            else
                                trainingdata[j][t] = "1";
                        }
                         if(t == 12){
                            if(trainingdata[j][t]<="47")trainingdata[j][t] = "0";
                            else
                                trainingdata[j][t] = "1";
                        }





                      
                    }

                
            }

              
    //TREE CREATION    
      

    attribute *root;
    set<int>temp;
cout<<endl<<"Please wait for around 20 seconds"<<endl;
      for(int i = 0; i<7; i++){
         root = new attribute();
         edges = 0;
        set<int>temp;
        ID3(&root,trainingdata,temp);
        treenodepointers.pb(root);
        cout<<i*3<<endl;
      
         
   }
   
int total = 0,ct = 0,bt = 0;
   for(int r = 0; r<treenodepointers.size();r++){
    total = 0;
    root = treenodepointers[r];
    

for(int i = 0; i<trainingdata.size();i++){
    if(trainingdata[i].size() == 15){
         int result = search(&root,trainingdata[i]);
            if(result == 1 && trainingdata[i][14] == ">50K")hit[r][i] = 1;
            else if(result == 0 && trainingdata[i][14] == ">50K")hit[r][i] = 0;
            else if(result == 1 && trainingdata[i][14] == "=<50K")hit[r][i] = 0;      
            else if(result == 0 && trainingdata[i][14] == "=<50K")hit[r][i] = 1;
        }
    }
}
   

   assign_weight();
  

    FILE *testingfile = fopen("test.txt","r+");
    while(fscanf(testingfile, "%[^\n]s",tempstore)!=EOF){
        int len = strlen(tempstore),j = 0;
        vector<string>strtemp;
    
        while(tempstore[j] == ' ')j++;
        int p = 0;
        string temp = "";
        for(;j<len;j++){
            if(tempstore[j] ==' ' || tempstore[j] ==','){
                if(temp!=""){
                    strtemp.pb(temp);  
                    p++;   
                }
                temp = "";
                continue;
            }
            temp.pb(tempstore[j]);
            
        }
        strtemp.pb(temp);
        testingdata.pb(strtemp);

        int lastinserted = testingdata.size()-1;
      
        for(int i = 0; i<testingdata[lastinserted].size();i++){
         string str = testingdata[lastinserted][i];
         if(str!="?")
          testdataset[i].insert(str);    
        }

     
       fgetc(testingfile);

    }

     for(int i = 0; i<15; i++){
                if( i == 0 || i == 2 || i == 4 || i == 10 || i == 11 || i == 12)continue;
                int len = dataset[i].size();
                int ct[len];
            for(set<string>::iterator it = dataset[i].begin();it!=dataset[i].end();it++){
               // int pos = 1;
                int pos = distance(dataset[i].begin(),it);//(it - dataset[i].begin());
               // pos--;
               
                  //  cout<<*it<<endl;
                    for(int t = 0; t<len ;t++)ct[t] = 0;
                       for(int j = 0; j<testingdata.size();j++){
                            if(testingdata[j].size() == 15 && testingdata[j][i] == *it)
                            ct[pos]++;
                        }
                }
            

                       int index = -1,maxvalue = -1;
                            for(int t = 0; t<len;t++){
                                    if(ct[t]>maxvalue){
                                        maxvalue = ct[t];
                                        index = t;
                                    }
                            }

                    string copy_string = "";
                    int val = 0;
                    for(set<string>::iterator it = dataset[i].begin();it!=dataset[i].end();it++){
                            if(val == index){
                                copy_string = *it;
                                break;
                            }
                            val++;
                    }
                   

                 for(int t = 0 ; t<testingdata.size();t++)

                        if(testingdata[t].size() == 15 && testingdata[t][i] == "?" ){
                          
                            testingdata[t][i] = "";
                            testingdata[t][i]+= copy_string;
                          
                        }
            
    }
    memset(arr,0,sizeof(arr));
    for(int i = 0; i<testingdata.size();i++){
        for(int t = 0; t<15; t++)
            if( (testingdata[i].size() == 15) && (t == 0 || t == 2 || t == 4 || t == 10 || t == 11 || t == 12)){
                string str = testingdata[i][t];
                if(str == "?")continue;
                char tempstring[100];
                for(int pp = 0; pp <str.size();pp++)
                    tempstring[pp] = str[pp];
                    int x = atoi(tempstring);
                    arr[t]+=x;
            }
    }
    int testingdatasize = testingdata.size();

    for(int i = 0; i<testingdatasize;i++){
        for(int t = 0; t<testingdata[i].size();i++)
           if( (testingdata[t].size() == 15) && (t == 0 || t == 2 || t == 4 || t == 10 || t == 11 || t == 12)){
               
            if(testingdata[i][t] == "?"){ 
                int temp = (arr[t]/testingdatasize);
                stringstream ss;
                ss<<temp;
                trainingdata[i][t] =ss.str();
          }
    }
    }

     for(int i = 0; i<15; i++){

        if( i ==0 || i == 2 || i == 4 || i == 10 || i == 11|| i == 12){
            testdataset[i].clear();
            testdataset[i].insert("0");
            testdataset[i].insert("1");
        }
        
    }   
 
    testdataset[0].insert("2");
    testdataset[0].insert("3");


    for(int j = 0; j<testingdata.size();j++){
                for(int t = 0; t<testingdata[j].size();t++){
                   if(testingdata[j].size()<15)continue;
                     if( t ==0 || t == 2 || t == 4 ||t == 10 || t == 11 || t == 12 ){
                        int mid = threshold[t].size()/2;
                            if(testingdata[j][t]<=threshold[t][mid])testingdata[j][t] = "0";
                            else
                                testingdata[j][t] = "1";
                        }
                    }
                    

                    
           }     
                
            
      
   
total = 0;
int ans = 0,inx;
ct = 0,bt = 0;
int wrong = 0,left = 0;

int test[20000][2];
double val[20000];
for(int i = 0; i<20000;i++)
test[i][0] = test[i][1] = 0;
ct = 0,bt = 0;

for(int r = 0; r<treenodepointers.size();r++){
   total = 0;
    root = treenodepointers[r];
    

for(int i = 0; i<testingdata.size();i++){


if(testingdata[i].size() == 15){
        int result = search(&root,testingdata[i]);
        if(result == 0)test[i][0]++;
        else
            test[i][1]++;        
    }
}

}

for(int i = 0; i<20000;i++)
val[i] = 0.0;
 ct = 0,bt = 0;
for(int r = 0; r<treenodepointers.size();r++){
   total = 0;
    root = treenodepointers[r];
    

for(int i = 0; i<testingdata.size();i++){


if(testingdata[i].size() == 15){
     if(test[i][0]>test[i][1])
            val[i]+= 1.0*classifier_weight[r]; 
        else
            val[i]-= 1.0*classifier_weight[r];

}

}


}
ct = 0,bt = 0;
total = 0;
for(int i = 0; i<testingdata.size();i++){
    if(testingdata[i].size()!=15)continue;
    total++;
 if(val[i]>0)ans = 1; else ans = 0;
       
        if(  ans == 1 && testingdata[i][14] == ">50K." ){ct++;
               cout<<">50K"<<"------------------"<<">50K"<<endl;
        }
        else if(ans == 0 && testingdata[i][14] == "<=50K."){
         bt++;
        cout<<"<=50K"<<"----------------"<<"<=50K"<<endl;
        }
        
        else if( ans == 0 && testingdata[i][14] == ">50K."){
         cout<<"<=50K"<<"xxxxxxxxxxxxxxxxxx"<<">50K"<<"   "<<endl;
        }

         else if( ans == 1 && testingdata[i][14] == "<=50K."){
         cout<<">50K"<<"xxxxxxxxxxxxxxxxxx"<<"<=50K"<<"    "<<endl;
        }    
}

for(int i = 0; i<7; i++)cout<<classifier_weight[i]<<"  ";

cout<<endl;
cout<<"Correctly Predicted:"<<ct+bt<<endl;
cout<<"Total Cases:        "<<total<<endl;
cout<<"----------------------------"<<endl;

cout<<"accuracy is:        "<<1.0*(ct+bt)/total*100<<endl;




    
    return 0;
}