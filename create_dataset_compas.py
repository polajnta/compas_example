import sqlite3
import pandas as pd
from dataclasses import dataclass
import logging
from collections import Counter, defaultdict
from dateutil.relativedelta import relativedelta
from datetime import datetime


from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")



logging.basicConfig(level=logging.INFO)
pd.set_option('display.max_columns', None) 
@dataclass
class Person:
    name: str
    id: int

def compare_people(x:Person, y:Person):
    return x.name == y.name and x.id == y.id

def import_data(db_path, skip_tables=['compas', 'summary']):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    sql_query = """SELECT name FROM sqlite_master  WHERE type='table';"""
    cur.execute(sql_query)
    tables = cur.fetchall()

    for table in tables:
        tableName = table[0]
        if tableName in skip_tables: continue
        yield tableName, pd.read_sql_query(f"SELECT * FROM {tableName}", con)
        
# process people df to drop compas info
def process_people(df):
    logging.info('People table, removing compass columns')
    logging.info(f'Before: {df.columns}')
    df.drop(df.iloc[:, 12:], inplace=True, axis=1)
    logging.info(f'After: {df.columns}')
    return df

def remove_compas_columns(df):
    logging.info('Removing compass columns')
    logging.info(f'Before: {df.columns}')
    cols = df.columns
    for c in cols:
        if 'compas' in c:
            df.drop(c, inplace=True, axis=1)
    logging.info(f'After: {df.columns}')
    return df
def process_word(w):
    if not w: return ''
    replacements = {v: ' ' for v in [' w/','.','<','>','/','-', ')', '(', '\\', ',']+[str(i) for i in range(10)]}
    replacements["'"] = 'ft'
    replacements['no '] = ['no_']
    w = w.lower()
    for r in replacements:
        w = w.replace(r, ' ')
    return w

def make_bow_text_index(textcols):  
    stop_words = ['an','of', 'to', 'without', 'while', 'with', 'on',  'none', 'at', 'by', 'as', 'the', 'or', 'in', 'over', 'and', 'nd', 'too', 'one', 'two', 'when', 'for']
    for col in textcols:
        values = col.unique()
        corpus = ' '.join([process_word(w) for w in filter(None,values)])
        counts = Counter(corpus.lower().split())
    words = [w for w in counts if counts[w]>4 and len(w)>1 and w not in stop_words]
    word_index = {j:i for i, j in enumerate(words)}
    return word_index


def days_in_custody(cust_df, start_date, end_date):
    if cust_df.empty: return 0
    
    #print(cust_df.in_custody, start_date)
    cust_df = cust_df[(cust_df.in_custody>=start_date) & (cust_df.out_custody <= end_date)]
    total_days = 0
    for _, c in cust_df.iterrows():
        # sometimes the in/out dates appear reversed for jail in particular
        total_days += abs((c.out_custody.to_pydatetime() - c.in_custody.to_pydatetime()).days)
    return total_days


def past_cases_vec(charges, prison, jail, words, date, max_priors=5):
    vec = {}
    charge_group = charges.groupby(['date_charge_filed'])
    # group words for all the past charges 
    # this is because there is so much variability in the way they are expressed
    for w in words:
        vec[f'past_{w}'] = 0
    vec['total_past_incidents'] = charge_group.ngroups
    vec['total_felonies_charges'] = charges[charges['charge_degree'].str.contains('\(F')].shape[0]
    vec['total_misdemeanors_charges'] = charges[charges['charge_degree'].str.contains('\(M')].shape[0]
    vec['total_other_cases'] = charges.shape[0] - vec['total_misdemeanors_charges'] - vec['total_felonies_charges']
    keys = list(charge_group.groups.keys())
    keys.sort()
    keys = keys[-max_priors:]

    i = 0 # charge group index 

    # for all the charges on a particular date
    ng = min(max_priors, len(keys))
    for k in keys:
        # label the cases in time-reverse order so 1 is most recent
        rev_ind = ng-i
    
        vec[f'{rev_ind}_felonies_case'] = 0
        vec[f'{rev_ind}_misdemeanors_case'] = 0
        vec[f'{rev_ind}_other_cases'] = 0
        cg = charge_group.get_group(k)
        if i+1 < len(keys):
            end_date = charge_group.get_group(keys[i+1]).iloc[0].date_charge_filed
        else:
            end_date = date
        
        
        vec[f'{rev_ind}_total_prison'] = days_in_custody(prison, k, end_date)
        vec[f'{rev_ind}_total_jail'] = days_in_custody(jail, k, end_date)

        # normalisation of days since last charge over the years
        # effects untested, assumend variable
        diff_days = (date.to_pydatetime() - k.to_pydatetime()).days
        #case_vec['days_since_current_offence'] = diff_days
        vec[f'{rev_ind}_time_norm'] = (1/2)**(diff_days/365)
        
        # for each charge in a group of charges
        for _, c in cg.iterrows():
            charge_words = process_word(c.charge).split()

            for w in charge_words:
                if w not in words: continue

                vec[f'past_{w}'] += 1

            if '(F' in c.charge_degree:
                vec[f'{rev_ind}_felonies_case'] += 1
            elif '(M' in c.charge_degree:
                vec[f'{rev_ind}_misdemeanors_case'] += 1
            else:
                vec[f'{rev_ind}_other_cases'] += 1

        
        i += 1
    
    return vec


def current_case_vec(person, cases, charges, date, words):
    case_vec = {}
    
    # person data
    case_vec['person_id'] = person.id
    case_vec['age'] = relativedelta(date, pd.to_datetime(person.dob)).years
    case_vec['race'] = person.race.lower()
    case_vec['sex'] = person.sex.lower()
    case_vec['j_felonies'] = person.juv_fel_count
    case_vec['j_misdemeanor'] = person.juv_misd_count
    case_vec['j_other'] = person.juv_other_count
    case_vec['felonies_case'] = 0
    case_vec['misdemeanors_case'] = 0
    case_vec['other_cases'] = 0
    case_vec['felonies_charge'] = 0
    case_vec['misdemeanors_charge'] = 0
    case_vec['other_charges']  = 0
    # make sure all words are in the case vec for df conv
    #for w in words:
    #    case_vec[w] = 0

    # arrest data
    for _, c in cases.iterrows():
        if '(F' in c.charge_degree:
            case_vec['felonies_case'] += 1
        elif '(M' in c.charge_degree:
            case_vec['misdemeanors_case'] += 1
        else:
            case_vec['other_cases'] += 1
    
    # charges 
    for w in words:
        case_vec[f'word_{w}'] = 0
    for _, c in charges.iterrows():
        if '(F' in c.charge_degree:
            case_vec['felonies_charge'] += 1
        elif '(M' in c.charge_degree:
            case_vec['misdemeanors_charge'] += 1
        else:
            case_vec['other_charges'] += 1

        charge_words = process_word(c.charge).split()
        for w in charge_words:
            if w in words:
                case_vec[f'word_{w}'] += 1

        

    return case_vec

def make_risk_pred_data_point(person, cases, charges, jail, prison, word_list, max_date, years=2, max_priors=5):
    cases = cases.sort_values('arrest_date', ascending=True)
    cases.arrest_date = pd.to_datetime(cases.arrest_date)
    charges.date_charge_filed = pd.to_datetime(charges.date_charge_filed)
    prison.in_custody = pd.to_datetime(prison.in_custody)
    prison.out_custody = pd.to_datetime(prison.out_custody)
    jail.in_custody = pd.to_datetime(jail.in_custody)
    jail.out_custody = pd.to_datetime(jail.out_custody)
    max_date_cutoff= max_date - relativedelta(years=years)
   
   
    case_group = cases.groupby(['case_number'])
    i = 0

    for _, cg in case_group:#cases.iterrows():
        c = cg.iloc[0]
    
        case_date = c.arrest_date

        # don't continue if the date is within years of dataset end
        if case_date > max_date_cutoff: break
        
        diff_date_future = case_date + relativedelta(years=years)
       
        prison_dates = prison[(prison.in_custody > case_date) & (prison.in_custody <= diff_date_future)]

        # skip instances where someone was in prison in the label period
        if not prison_dates.empty:  continue 

        current_charges = charges[charges.case_number == c.case_number]
        #previous_cases = cases[cases.arrest_date < case_date]
        previous_charges = charges[charges.date_charge_filed < case_date]
        previous_prison = prison[prison.in_custody < case_date] #assuming they are out of prison at the time of new charge
        i += 1

        previous_jail = jail[jail.in_custody < case_date] 
        future_charges = charges[(charges.case_number != c.case_number) & (charges.date_charge_filed > case_date) & (charges.date_charge_filed <= diff_date_future)] 
        
        current_case = current_case_vec(person, cg, current_charges, case_date, word_list)
    
        # felony in the next y years
        current_case['pred_label'] = 0 if future_charges[future_charges['charge_degree'].str.contains('\(F')].empty else 1
        
        # commit a sexual offence in the next y years
        #current_case['sex_crime_label'] = 0 if future_charges[future_charges['charge'].str.contains('Sex',na=False)].empty else 1
        
        # commit a murder in the next y years
        #current_case['murder_label'] = 0 if future_charges[future_charges['charge'].str.contains('Murder',na=False)].empty else 1

        past_vec = past_cases_vec(previous_charges, previous_prison, previous_jail, word_list, case_date, max_priors=max_priors)
        
        
        yield {**current_case, **past_vec}
 
        
def join_tabular_compas_for_risk(db_path, years=1):

    data = []
    dataframes = { t:df for t, df in import_data(db_path) }
    
    print(dataframes.keys())
    i = 0 
 
    process_people(dataframes['people'])
    remove_compas_columns(dataframes['casearrest'])
    remove_compas_columns(dataframes['charge'])
    

    ca_df = dataframes['casearrest']
    ch_df = dataframes['charge']
    ja_df = dataframes['jailhistory']
    pr_df = dataframes['prisonhistory']
    word_list = make_bow_text_index([ch_df['charge']])
    
    max_date = datetime.strptime('2016-03-30', '%Y-%m-%d') #eyeballed from case arrest_date

    for ind, p in dataframes['people'].iterrows():
        print('.', end='', flush=True)
        person_cases = ca_df.loc[ca_df['person_id'].isin([p['id']])]
        person_charges = ch_df.loc[ch_df['person_id'].isin([p['id']])]
        person_jail = ja_df.loc[ja_df['person_id'].isin([p['id']])]
        person_prison = pr_df.loc[pr_df['person_id'].isin([p['id']])]
        person_name = [p['name']]
        
        # check data frames align on person name
        for name_frame in [person_cases, person_charges]:
            if name_frame.empty: continue
            person_name.append(name_frame['name'].iloc[0])

        for name_frame in [person_jail, person_prison]:
            if name_frame.empty: continue
            person_name.append(name_frame['first'].iloc[0]+' '+name_frame['last'].iloc[0])
        
        # if the set of names has more than value in it the names did not align
        # skip this person
        person_name = set(person_name)
        if len(person_name) > 1:
            logging.info('person does not match', person_name)
            continue
        
        if not person_cases.empty:
            i += 1

            for row in make_risk_pred_data_point(p, person_cases, person_charges, person_jail, person_prison, word_list, max_date, years=years):
                data.append(row)
        

    return pd.DataFrame(data)

def split_dataset(df, val=True):
    pids = df.person_id.unique()
    train, test = train_test_split(pids, test_size=0.2,random_state=42, shuffle=True)
    val_df = None

    train, val = train_test_split(train, test_size=0.25, random_state=1, shuffle=True)
    val_df = df[df.person_id.isin(val)]
    

    train_df = df[df.person_id.isin(train)]
    test_df = df[df.person_id.isin(test)]
    print(train_df.shape, test_df.shape, val_df.shape)
    return train_df, test_df, val_df

def undersample_by_column(train, column, value, same_as_value):
    logging.info('Undersampling by', column)
    sample_size = train[train[column]==same_as_value].shape[0]
    selected = train[train[column]==value]
    selected = selected.sample(n=(selected.shape[0]-sample_size))
    selected = train.drop(index = selected.index)
    logging.info('People before:',train['person_id'].unique().shape[0])
    logging.info('People after:', selected['person_id'].unique().shape[0])
    logging.info('Unique before:', train[column].value_counts())
    logging.info('Unique after:', selected[column].value_counts())
    return selected

if __name__ == "__main__":
    outpath = 'joined_data/compas_v2_all_1y-label.pqt'
    data = join_tabular_compas_for_risk('compas.db')
    data.to_parquet(outpath)

 