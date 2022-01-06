import pandas as pd

excelFile = pd.read_excel('자동매매보고서.xlsx')
temp = [{'Coin' : 'testcoing', '금액' : '1000원', '현재금액' : '2000원'}]
excelFile = excelFile.append(temp)
excelFile.to_excel(excelFile, index=False)
excelFile.save()

#%%
f = open('C:/Users/LeeKihoon/Desktop/test.txt', 'a')

f.write("구매금액 : ")
f.write("100원 \t")
f.write("판매금액 : ")
f.write("100원 \n")
f.write("구매금액 : ")
f.write("100원 \t")
f.write("판매금액 : ")
f.write("100원 \n")


f.close()


