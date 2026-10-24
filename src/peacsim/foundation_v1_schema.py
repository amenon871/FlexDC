"""Foundation receipt schema boundaries; missing provenance is not a behavior label."""
import json
import math
import numpy as np
import pandas as pd

# Mirrors the existing PLAN_PROVENANCE_COLUMNS mapping, without changing its order.
PROVENANCE_KEYS = {
    'Plan_Row_ID':'plan_row_id', 'Base_Plan_Row_ID':'base_plan_row_id',
    'Design_Version':'design_version', 'Context_ID':'context_id', 'PR_ID':'pr_id',
    'PR_Stratum':'pr_stratum', 'PR_Stratum_Detail':'pr_stratum_detail',
    'Weight_ID':'weight_id', 'Weight_Stratum':'weight_stratum',
    'Weight_Family':'weight_family', 'Weight_Detail':'weight_detail',
    'Dirichlet_Alpha':'dirichlet_alpha', 'Anchor_Type':'anchor_type',
    'Data_Split':'data_split', 'R_Over_P':'R_over_P',
    'Configured_R_Over_P_Max':'configured_r_over_p_max',
    'Configured_Weight_Lower':'configured_weight_lower',
    'Configured_Weight_Upper':'configured_weight_upper',
    'Simulation_Seed':'simulation_seed', 'Is_Replicate':'is_replicate',
    'Replicate_Of_Plan_Row_ID':'replicate_of_plan_row_id', 'Plan_Source_File':'plan_source_file',
}
REQUIRED_PROVENANCE = frozenset(('Plan_Row_ID','Base_Plan_Row_ID','Design_Version',
    'Context_ID','Data_Split','Simulation_Seed','Is_Replicate'))
OPTIONAL_PROVENANCE = frozenset(PROVENANCE_KEYS) - REQUIRED_PROVENANCE
OPTIONAL_DIAGNOSTICS = frozenset(('PR_Sweep_Mode','Pmin_kw_per_server','Pmax_kw_per_server',
    'Pbar_lower_bound_kw_per_server','Pbar_upper_bound_kw_per_server','PR_upper_bound_kw_per_server',
    'R_lower_bound_kw_per_server','R_max_for_this_Pbar_kw_per_server','Pbar_points','R_points_per_Pbar',
    'PR_Total_Pairs_Before_Chunk','PR_Chunk_Index','PR_Num_Chunks','PR_Chunk_Start_Index',
    'PR_Chunk_End_Index_Exclusive','PR_Pairs_In_Chunk'))


class InvalidFoundationObservation(ValueError):
    """Controlled technical rejection; never a valid/feasible scientific label."""


def missing_scalar(value):
    if value is None:
        return True
    if isinstance(value,str):
        return value.strip().lower() in ('','nan','none','nat','<na>')
    if isinstance(value,(list,tuple,dict,np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError,ValueError):
        return False


def optional_scalar(value):
    """CSV missing representation for optional scalar provenance is always blank."""
    if missing_scalar(value):
        return ''
    if isinstance(value,(dict,list,tuple,np.ndarray)):
        raise ValueError('Expected scalar provenance, got '+type(value).__name__)
    if hasattr(value,'item'):
        value=value.item()
    if isinstance(value,float) and not math.isfinite(value):
        return ''
    return value


def normalize_metadata(metadata):
    row=dict(metadata)
    for column,key in PROVENANCE_KEYS.items():
        value=row.get(key,row.get('r_over_p','') if column=='R_Over_P' else '')
        if column in REQUIRED_PROVENANCE:
            if missing_scalar(value):
                raise ValueError('Missing required Foundation provenance: '+column+' ('+key+')')
            if column=='Simulation_Seed':
                finite_number(value,'plan_metadata.'+key)
            elif isinstance(value,float) and not math.isfinite(value):
                raise ValueError('Invalid required Foundation provenance: '+column)
            row[key]=value.item() if hasattr(value,'item') else value
        else:
            row[key]=optional_scalar(value)
    for key in ('Pbar_kw_per_server','R_kw_per_server','server_count','utilization','job_type_count'):
        if key in row:finite_number(row[key],'plan_metadata.'+key)
    if 'weights' in row:
        weights=json.loads(row['weights']) if isinstance(row['weights'],str) else row['weights']
        if not isinstance(weights,(list,tuple,np.ndarray)):
            raise ValueError('Invalid required plan_metadata.weights')
        for i,value in enumerate(weights):finite_number(value,'plan_metadata.weights['+str(i)+']')
    flag=row['is_replicate']
    if str(flag).lower() not in ('true','false','1','0','yes','no','y','n','t','f'):
        raise ValueError('Invalid required Foundation provenance: Is_Replicate')
    repeat=str(flag).lower() in ('true','1','yes','y','t')
    if repeat and not row['replicate_of_plan_row_id']:
        raise ValueError('Missing repeat parent: Replicate_Of_Plan_Row_ID')
    return row


def finite_number(value,path):
    try:
        number=float(value)
    except (ValueError,TypeError,OverflowError) as error:
        raise InvalidFoundationObservation('INVALID_OBSERVATION: required numeric field '+path+' is missing/non-numeric') from error
    if not math.isfinite(number):
        raise InvalidFoundationObservation('INVALID_OBSERVATION: required numeric field '+path+' is nonfinite')
    return number


def validate_required_labels(results,diagnostics,job_count):
    """Shared by the writer and canonical audit; same existing required labels."""
    for label,record in (('results',results),('diagnostics',diagnostics)):
        for key in ('Pbar_kw_per_server','R_kw_per_server','server_count','utilization','Simulation_Seed'):
            if key not in record:
                raise InvalidFoundationObservation('INVALID_OBSERVATION: missing '+label+'.'+key)
            finite_number(record[key],label+'.'+key)
    for key in ('Mtrack_Error_MeanAbs_Normalized','Simulator_RSR_Total_Cost'):
        finite_number(results.get(key),'results.'+key)
    for key in ('Ctrack_Epsilon_90th','Diagnostic_FullPaperObjective_Cost'):
        finite_number(diagnostics.get(key),'diagnostics.'+key)
    for i in range(job_count):
        finite_number(results.get('Weight_'+str(i)),'results.Weight_'+str(i))
    raw=results.get('QoS_Delay_Probabilities')
    def invalid_constant(token):
        raise InvalidFoundationObservation('INVALID_OBSERVATION: required results.QoS_Delay_Probabilities contains '+token)
    try:
        pj=json.loads(raw,parse_constant=invalid_constant) if isinstance(raw,str) else raw
    except (ValueError,TypeError) as error:
        if isinstance(error,InvalidFoundationObservation):raise
        raise InvalidFoundationObservation('INVALID_OBSERVATION: results.QoS_Delay_Probabilities is invalid JSON') from error
    if not isinstance(pj,(list,tuple)) or len(pj)!=job_count:
        raise InvalidFoundationObservation('INVALID_OBSERVATION: results.QoS_Delay_Probabilities width must equal J='+str(job_count))
    for i,value in enumerate(pj):
        finite_number(value,'results.QoS_Delay_Probabilities['+str(i)+']')


def json_safe(value,path='receipt',optional=False):
    """Recursive strict-JSON conversion. Only declared optional fields may vanish."""
    if isinstance(value,dict):
        return {str(k):json_safe(v,path+'.'+str(k),optional) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [json_safe(v,path+'['+str(i)+']',optional) for i,v in enumerate(value)]
    if missing_scalar(value):
        if optional:return None
        # Blank strings are legitimate serialized optional/header values; actual
        # required identities/labels have already been checked separately.
        if value is None or isinstance(value,str):return value
        raise InvalidFoundationObservation('INVALID_OBSERVATION: nonfinite/missing value at '+path)
    if hasattr(value,'item'):
        return json_safe(value.item(),path,optional)
    if isinstance(value,float) and not math.isfinite(value):
        if optional:return None
        raise InvalidFoundationObservation('INVALID_OBSERVATION: nonfinite value at '+path)
    if isinstance(value,(str,int,float,bool)) or value is None:
        return value
    raise InvalidFoundationObservation('INVALID_OBSERVATION: unsupported '+type(value).__name__+' at '+path)


def safe_record(record,label):
    out={}
    for key,value in record.items():
        if key in REQUIRED_PROVENANCE and missing_scalar(value):
            raise InvalidFoundationObservation('INVALID_OBSERVATION: required provenance '+label+'.'+key+' is missing')
        if key in OPTIONAL_PROVENANCE:
            value=optional_scalar(value)
        out[key]=json_safe(value,label+'.'+key,optional=label=='diagnostics' and key in OPTIONAL_DIAGNOSTICS)
    return out
