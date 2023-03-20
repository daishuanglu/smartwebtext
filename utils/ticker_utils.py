from fastDamerauLevenshtein import damerauLevenshtein
import pandas as pd
import re

ticlutbl=pd.read_csv("utils/ais_ticker.csv")

common_ext=[' company'," inc"," corp",' ltd','group ',' co',' llc',' plc',' limited']

def standardize(s):
    s=s.lower()
    for i in common_ext:
        s = s.replace(i, '')
    s = ' '.join(re.sub(r'[^a-zA-Z0-9]', ' ', s.rstrip()).split())
    return s

ticlutbl["Company"]=ticlutbl["Company"].apply(lambda x: standardize(re.sub("[\(\[].*?[\)\]]", "",x.lower())))


def ticker_finder(company, verbose=True):
    s =  standardize(company)
    if verbose: print("input company:", s)
    max_sim=0
    id=-1
    for i in range(len(ticlutbl)):
        sim=damerauLevenshtein(s.split(),ticlutbl["Company"][i].split(), similarity=True)
        #print(i,sim)
        if sim>max_sim:
            max_sim=sim
            id=i

    #if max_sim==0.5:
    #    if s.split()[0]!= ticlutbl["Company"][id].split()[0]:
    #        return
    if max_sim<0.5:
        return
    if verbose: print("similarity: ",max_sim, ", find company:",ticlutbl["Company"][id])
    return ticlutbl["ticker"][id]


def main():

    test_companies=pd.read_csv("company_actions.csv")['Company'].tolist()
    #test_companies=["columbia bank"]
    nfound=0
    for comp in test_companies:
        tic=ticker_finder(comp)
        if tic:
            nfound+=1
        print(tic )
    print(nfound, len(test_companies))
    return

if __name__=="__main__":
    main()


