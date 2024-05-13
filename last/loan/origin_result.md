
(venv) PS C:\Users\yegetables\PycharmProjects\pythonProject> time python -u "c:\Users\yegetables\PycharmProjects\pythonProject\last\loan\loan.py"
load data,cache=False,isPart=True
read  C:\Users\yegetables\PycharmProjects\pythonProject\last\loan/data/train_part.csv (10000, 47)
训练ID3模型 start
训练ID3模型 success
zero_rank= 0.7975
决策树：ID3,预测测试集结果如下：
The accuracy of the validation on the training data set is : 1.0
The accuracy of the validation on the test data set is : 0.704
均方误差：0.296
平均绝对误差：0.296
Feature: loanAmnt, Importance: 0.03363984946538531
Feature: term, Importance: 0.007186206275937316
Feature: interestRate, Importance: 0.02870162590096264
Feature: installment, Importance: 0.05235031228672011
Feature: grade, Importance: 0.0006435462534847729
Feature: subGrade, Importance: 0.08999665517546762
Feature: employmentTitle, Importance: 0.05406241078642334
Feature: employmentLength, Importance: 0.015261387849299401
Feature: homeOwnership, Importance: 0.0067040699175411654
Feature: annualIncome, Importance: 0.033560674574999595
Feature: verificationStatus, Importance: 0.004330947752449111
Feature: issueDate, Importance: 0.0510234826239005
Feature: purpose, Importance: 0.006941318392502058
Feature: postCode, Importance: 0.06374462744608037
Feature: regionCode, Importance: 0.030345347112702984
Feature: dti, Importance: 0.05591901873946108
Feature: delinquency_2years, Importance: 0.010740438638995016
Feature: ficoRangeLow, Importance: 0.01657655817806016
Feature: ficoRangeHigh, Importance: 0.025152876327153885
Feature: openAcc, Importance: 0.014300025696994446
Feature: pubRec, Importance: 0.0015206444983694258
Feature: pubRecBankruptcies, Importance: 0.007690816887215895
Feature: revolBal, Importance: 0.042660037682675324
Feature: revolUtil, Importance: 0.0463619246142997
Feature: totalAcc, Importance: 0.03413305984794393
Feature: initialListStatus, Importance: 0.0038596310840906044
Feature: applicationType, Importance: 0.0012920502580659385
Feature: earliesCreditLine, Importance: 0.05894307693473989
Feature: title, Importance: 0.018307932305465697
Feature: n0, Importance: 0.003146273877395916
Feature: n1, Importance: 0.013436400271061766
Feature: n2, Importance: 0.00937740850862812
Feature: n3, Importance: 0.005596375676050466
Feature: n4, Importance: 0.013943769270171952
Feature: n5, Importance: 0.024914315652087453
Feature: n6, Importance: 0.02997159027734142
Feature: n7, Importance: 0.019540612586382046
Feature: n8, Importance: 0.02336720773841149
Feature: n9, Importance: 0.007983819179267233
Feature: n10, Importance: 0.011343757981534128
Feature: n11, Importance: 0.0
Feature: n12, Importance: 0.0
Feature: n13, Importance: 0.0007663926609942135
Feature: n14, Importance: 0.02066152281328667
训练CART模型 start
训练CART模型 success
zero_rank= 0.7975
决策树：CART,预测测试集结果如下：
The accuracy of the validation on the training data set is : 1.0
The accuracy of the validation on the test data set is : 0.7045
均方误差：0.2955
平均绝对误差：0.2955
Feature: loanAmnt, Importance: 0.031975159765089495
Feature: term, Importance: 0.005119375081865707
Feature: interestRate, Importance: 0.030913641709380667
Feature: installment, Importance: 0.041663141329593284
Feature: grade, Importance: 0.0014536967575227145
Feature: subGrade, Importance: 0.09119226705029404
Feature: employmentTitle, Importance: 0.05935126669762842
Feature: employmentLength, Importance: 0.019688265433523394
Feature: homeOwnership, Importance: 0.011692148907572156
Feature: annualIncome, Importance: 0.03604368780838407
Feature: verificationStatus, Importance: 0.007740809661289784
Feature: issueDate, Importance: 0.05026612921380869
Feature: purpose, Importance: 0.009553405675100062
Feature: postCode, Importance: 0.046222693319720815
Feature: regionCode, Importance: 0.03774288131933757
Feature: dti, Importance: 0.04770970571561888
Feature: delinquency_2years, Importance: 0.008061902857198934
Feature: ficoRangeLow, Importance: 0.013934192097653774
Feature: ficoRangeHigh, Importance: 0.01511457899135642
Feature: openAcc, Importance: 0.013027067563713532
Feature: pubRec, Importance: 0.006797029790119779
Feature: pubRecBankruptcies, Importance: 0.0029856824883365313
Feature: revolBal, Importance: 0.05431593605202357
Feature: revolUtil, Importance: 0.03638332138412112
Feature: totalAcc, Importance: 0.03814051374391627
Feature: initialListStatus, Importance: 0.0029973186437880803
Feature: applicationType, Importance: 0.0021473121011971632
Feature: earliesCreditLine, Importance: 0.06478219484264855
Feature: title, Importance: 0.011946789782310226
Feature: n0, Importance: 0.011199387608325439
Feature: n1, Importance: 0.01035789160539654
Feature: n2, Importance: 0.0036144847558760946
Feature: n3, Importance: 0.007684583927963482
Feature: n4, Importance: 0.016474107980735044
Feature: n5, Importance: 0.031139430985357132
Feature: n6, Importance: 0.0300859580916443
Feature: n7, Importance: 0.010622123993171027
Feature: n8, Importance: 0.028283437252216114
Feature: n9, Importance: 0.01333860911844858
Feature: n10, Importance: 0.016617439981919366
Feature: n11, Importance: 0.0
Feature: n12, Importance: 0.0007055097107610232
Feature: n13, Importance: 0.0
Feature: n14, Importance: 0.020914919204072232
训练C45模型 start
给定第(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43)特征为连续值特征
执行代价复杂度剪枝，惩罚参数：α = 2.5
剪枝前，C4.5决策树的叶结点总数：32
剪枝后，C4.5决策树的叶结点总数：31
训练C45模型 success
zero_rank= 0.7975
决策树：C45,预测测试集结果如下：
The accuracy of the validation on the training data set is : 0.81025
The accuracy of the validation on the test data set is : 0.797
均方误差：0.203
平均绝对误差：0.203
real    2m 16.02s
user    0m 0.00s
sys     0m 0.01s






(venv) PS C:\Users\yegetables\PycharmProjects\pythonProject> time python -u "c:\Users\yegetables\PycharmProjects\pythonProject\last\loan\loan.py"
load data,cache=False,isPart=True
read  C:\Users\yegetables\PycharmProjects\pythonProject\last\loan/data/train_part.csv (100000, 47)
训练ID3模型 start
训练ID3模型 success
zero_rank= 0.8032
决策树：ID3,预测测试集结果如下：
The accuracy of the validation on the training data set is : 1.0
The accuracy of the validation on the test data set is : 0.7039
均方误差：0.2961
平均绝对误差：0.2961
Feature: loanAmnt, Importance: 0.02553042770380391
Feature: term, Importance: 0.005422991703753209
Feature: interestRate, Importance: 0.030282117091211723
Feature: installment, Importance: 0.040385592619054715
Feature: grade, Importance: 0.001715128207796966
Feature: subGrade, Importance: 0.08086778585170923
Feature: employmentTitle, Importance: 0.05399936593552366
Feature: employmentLength, Importance: 0.021261233506249706
Feature: homeOwnership, Importance: 0.009645122425561732
Feature: annualIncome, Importance: 0.04391892809751422
Feature: verificationStatus, Importance: 0.009787133239626222
Feature: issueDate, Importance: 0.04059160253406018
Feature: purpose, Importance: 0.008860616901110925
Feature: postCode, Importance: 0.0586366781514288
Feature: regionCode, Importance: 0.03824447720837656
Feature: dti, Importance: 0.05716402569464043
Feature: delinquency_2years, Importance: 0.008114796144854537
Feature: ficoRangeLow, Importance: 0.017380099461879215
Feature: ficoRangeHigh, Importance: 0.017429944663546748
Feature: openAcc, Importance: 0.016171965780310778
Feature: pubRec, Importance: 0.005969043477800913
Feature: pubRecBankruptcies, Importance: 0.003925876880863195
Feature: revolBal, Importance: 0.05321026130204802
Feature: revolUtil, Importance: 0.04801060692383816
Feature: totalAcc, Importance: 0.030865116797059323
Feature: initialListStatus, Importance: 0.00425511165633295
Feature: applicationType, Importance: 0.0007580235028194676
Feature: earliesCreditLine, Importance: 0.05208188470630286
Feature: title, Importance: 0.015635668149741135
Feature: n0, Importance: 0.009415289669195734
Feature: n1, Importance: 0.014642756742514147
Feature: n2, Importance: 0.008968843221519095
Feature: n3, Importance: 0.010139957832833097
Feature: n4, Importance: 0.0156908946377929
Feature: n5, Importance: 0.02516250360868455
Feature: n6, Importance: 0.028910290918757874
Feature: n7, Importance: 0.016045103572214024
Feature: n8, Importance: 0.02739218465256399
Feature: n9, Importance: 0.009860991737270485
Feature: n10, Importance: 0.014976623424385795
Feature: n11, Importance: 0.00010807019027746182
Feature: n12, Importance: 0.00014007974934932474
Feature: n13, Importance: 0.0017825006259839444
Feature: n14, Importance: 0.016642283097838127
训练CART模型 start
训练CART模型 success
zero_rank= 0.8032
决策树：CART,预测测试集结果如下：
The accuracy of the validation on the training data set is : 1.0
The accuracy of the validation on the test data set is : 0.7029
均方误差：0.2971
平均绝对误差：0.2971
Feature: loanAmnt, Importance: 0.022273663693431252
Feature: term, Importance: 0.005951980828813332
Feature: interestRate, Importance: 0.027909295116111198
Feature: installment, Importance: 0.042733836979016726
Feature: grade, Importance: 0.0016753869536197414
Feature: subGrade, Importance: 0.08030998250138265
Feature: employmentTitle, Importance: 0.05120529950089538
Feature: employmentLength, Importance: 0.019773547845505047
Feature: homeOwnership, Importance: 0.010862006637347375
Feature: annualIncome, Importance: 0.044452188824254155
Feature: verificationStatus, Importance: 0.008121532394796457
Feature: issueDate, Importance: 0.04008234093609615
Feature: purpose, Importance: 0.010273766685017232
Feature: postCode, Importance: 0.049797193351639096
Feature: regionCode, Importance: 0.03683856933151321
Feature: dti, Importance: 0.057277771500077575
Feature: delinquency_2years, Importance: 0.008842354350460876
Feature: ficoRangeLow, Importance: 0.015956321005969772
Feature: ficoRangeHigh, Importance: 0.017469172677546833
Feature: openAcc, Importance: 0.016455338586923936
Feature: pubRec, Importance: 0.007206104508082535
Feature: pubRecBankruptcies, Importance: 0.0042481551151394336
Feature: revolBal, Importance: 0.05154532350861858
Feature: revolUtil, Importance: 0.053018074142314464
Feature: totalAcc, Importance: 0.03310769453543926
Feature: initialListStatus, Importance: 0.004875348803445947
Feature: applicationType, Importance: 0.001342344925542119
Feature: earliesCreditLine, Importance: 0.052922019140385676
Feature: title, Importance: 0.01559849060145638
Feature: n0, Importance: 0.011610395290745186
Feature: n1, Importance: 0.014161704610901712
Feature: n2, Importance: 0.008918615515146763
Feature: n3, Importance: 0.011349787401072883
Feature: n4, Importance: 0.01759403078103287
Feature: n5, Importance: 0.02451640586764085
Feature: n6, Importance: 0.03024047352160623
Feature: n7, Importance: 0.01810351136484635
Feature: n8, Importance: 0.022674803603337106
Feature: n9, Importance: 0.011284212352243568
Feature: n10, Importance: 0.015164118460307794
Feature: n11, Importance: 6.856134927509792e-05
Feature: n12, Importance: 0.00046874921675404757
Feature: n13, Importance: 0.0021715648621279645
Feature: n14, Importance: 0.019547960822119053
训练C45模型 start
给定第(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43)特征为连续值特征
执行代价复杂度剪枝，惩罚参数：α = 2.5
剪枝前，C4.5决策树的叶结点总数：92
剪枝后，C4.5决策树的叶结点总数：88
训练C45模型 success
zero_rank= 0.8032
决策树：C45,预测测试集结果如下：
The accuracy of the validation on the training data set is : 0.799475
The accuracy of the validation on the test data set is : 0.80115
均方误差：0.19885
平均绝对误差：0.19885
real    1h 33m 48s
user    0m 0.00s
sys     0m 0.01s