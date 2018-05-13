#计算KS并画图
def evaluate_performance(all_target, predicted, toplot=True):
    fpr, tpr, thresholds = roc_curve(all_target, predicted)
    roc_auc = auc(fpr, tpr)
    ks = max(tpr-fpr)
    maxind = plb.find(tpr-fpr == ks)

    #?
    event_rate = sum(all_target) / 1.0 / all_target.shape[0]
    cum_total = tpr * event_rate + fpr * (1-event_rate)
    minind = plb.find(abs(cum_total - event_rate) == min(abs(cum_total - event_rate)))
    if minind.shape[0] > 0:
        minind = minind[0]

    print ('KS=' + str(round(ks, 3)) + ', AUC=' + str(round(roc_auc,2)) +', N='+str(predicted.shape[0]))
    print ('At threshold=' + str(round(event_rate, 3)) + ', TPR=' + str(round(tpr[minind],2)) + ', ' + str(int(round(tpr[minind]*event_rate*all_target.shape[0]))) + ' out of ' + str(int(round(event_rate*all_target.shape[0]))))
    print ('At threshold=' + str(round(event_rate, 3)) + ', FPR=' + str(round(fpr[minind],2)) + ', ' + str(int(round(fpr[minind]*(1.0-event_rate)*all_target.shape[0]))) + ' out of ' + str(int(round((1.0-event_rate)*all_target.shape[0]))))  
    
    # Score average by percentile
    binnum = 10
    ave_predict = np.zeros((binnum))
    ave_target = np.zeros((binnum))
    indices = np.argsort(predicted)
    binsize = int(round(predicted.shape[0]/1.0/binnum))
    for i in list(range(binnum)):
        startind = i*binsize
        endind = min(predicted.shape[0], (i+1)*binsize)
        ave_predict[i] = np.mean(predicted[indices[startind:endind]])
        ave_target[i] = np.mean(all_target[indices[startind:endind]])
    print ('Ave_target: ' + str(ave_target))
    print ('Ave_predicted: ' + str(ave_predict))
    
    if toplot:
        # KS plot
        plt.figure(figsize=(20,6))
        plt.subplot(1,3,1)
        plt.plot(fpr, tpr)
        plt.hold
        plt.plot([0,1],[0,1], color='k', linestyle='--', linewidth=2)
        plt.title('KS='+str(round(ks,2))+ ' AUC='+str(round(roc_auc,2)), fontsize=20)
        plt.plot([fpr[maxind], fpr[maxind]], [fpr[maxind], tpr[maxind]], linewidth=4, color='r')
        plt.plot([fpr[minind]], [tpr[minind]], 'k.', markersize=10)

        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('False positive', fontsize=20); plt.ylabel('True positive', fontsize=20);
        
        #print 'At threshold=' + str(round(event_rate, 3))
        #print str(round(fpr[minind],2))
        #print str(int(round(fpr[minind]*(1.0-event_rate)*all_target.shape[0])))
        #print str(int(round((1.0-event_rate)*all_target.shape[0]))) 
        
    
        # Score distribution score
        plt.subplot(1,3,2)
        #print predicted.columns
        plt.hist(predicted, bins=20)
        plt.hold
        plt.axvline(x=np.mean(predicted), linestyle='--')
        plt.axvline(x=np.mean(all_target), linestyle='--', color='g')
        plt.title('N='+str(all_target.shape[0])+' Tru='+str(round(np.mean(all_target),3))+' Pred='+str(round(np.mean(predicted),3)), fontsize=20)
        plt.xlabel('Target rate', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        
        plt.subplot(1,3,3)
        plt.plot(ave_predict, 'b.-', label='Prediction', markersize=5)
        plt.hold
        plt.plot(ave_target, 'r.-', label='Truth', markersize=5)
        plt.legend(loc='lower right')
        plt.xlabel('Percentile', fontsize=20)
        plt.ylabel('Target rate', fontsize=20)
        
        plt.show()

    return ks

#调用函数
evaluate_performance(train_Y, predict_Y,toplot=True)



#pi1为样本bad_rate
def Lift_Gains(fpr,tpr,pi1):
    Ptp=pi1*tpr
    Pfp=(1-pi1)*fpr
    Depth=Ptp+Pfp
    PV_plus=Ptp/Depth
    Lift=PV_plus/pi1
  
    return Depth,PV_plus,Lift

#调用函数
#LIft图
plt.figure(figsize=(30,6))
plt.subplot(1,3,1)
#设置x轴范围
plt.xlim(0,1)
plt.ylim(1,3)
plt.title('LIFT')
plt.plot(Depth, Lift)

#Gains图
plt.subplot(1,3,2)
#设置x轴范围
plt.xlim(0,1)
plt.ylim(0.05,0.6)
plt.title('Gains')
plt.plot(Depth, PV_plus,color="m",linestyle="-")
plt.show() 


#画KS图
def KS_calculation_plot(y, y_pred, bin_num=20):
    '''
    Calculate KS
    '''
    df = pd.DataFrame()
    df['y'] = y
    df['y_pred'] = y_pred
    n_sample = df.shape[0]
    y_cnt = df['y'].sum()
    #向上取整
    bucket_sample = ceil(n_sample/bin_num)
    #df按y_pred进行排序
    df = df.sort_values('y_pred', ascending = False)
    df['group'] = [ceil(x/bucket_sample) for x in range(1, n_sample+1)]

    grouped = df.groupby('group')['y'].agg({'Totalcnt': 'count',
                                            'Y_rate': np.mean,
                                            'Y_pct': lambda x: np.sum(x / y_cnt),
                                            'n_Y_pct': lambda x: np.sum((1-x) / (n_sample-y_cnt))})
    grouped['Cum_Y_pct'] = grouped['Y_pct'].cumsum()
    grouped['Cum_nY_pct'] = grouped['n_Y_pct'].cumsum()
    grouped['KS'] = (grouped['Cum_Y_pct'] - grouped['Cum_nY_pct']).map(lambda x: abs(x))

    KS = grouped['KS'].max()
    
    plt.figure(figsize=(20,6))
    plt.subplot(1,3,1)
    plt.plot(grouped['Cum_Y_pct'])
    plt.hold
    plt.plot(grouped['Cum_nY_pct'])
    plt.show()
    return KS

#调用函数
KS_calculation_plot(train_Y,predict_Y, 40)