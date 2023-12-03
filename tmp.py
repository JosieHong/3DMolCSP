import json
import re

def load_json_with_check(log_content):
    search_res = re.search(r'"BranchBlock": \[(\s|.)*?\],', log_content)
    if search_res != None: 
        unsolvable_string = log_content[search_res.start(): search_res.end()].rstrip(',')
    
        branch_block = unsolvable_string[16:-1]
        branch_block = ''.join(branch_block.split()).split('",')
        branch_block = [i[1:] for i in branch_block] # remove the stat " in each item
        branch_block[-1] = branch_block[-1][:-1] # remove the end " in the last item
    
        log_content = log_content[:search_res.start()] + log_content[search_res.end():]
        json_dict = json.loads(log_content)
        json_dict['BranchBlock'] = branch_block
    
    else:
        json_dict = json.loads(log_content)
    return json_dict

json_string1 = """{
    "CallerSdkName": "Ironsource",
    "ConditionBlock": [
        "if $r3 != $r2 goto $r5 = staticinvoke <java.lang.Boolean: java.lang.Boolean valueOf(boolean)>($z0)"
    ],
    "BranchBlock": [
        "staticinvoke <com.vungle.warren.Vungle: void updateConsentStatus(com.vungle.warren.Vungle$Consent,java.lang.String)>($r4, \"1.0.0\")",
        "setence 2",
        "setence 3"
        ],
    "PackageName": "com.TecnoveApp.Divulgacion",
    "ApiType": "GDPR",
    "FlawType": "ConditionalEnforcementForWrapper",
    "DelegationPath": [
        "<com.ironsource.mediationsdk.IronSource: void setConsent(boolean)>",
        "<com.ironsource.mediationsdk.IronSourceObject: void setConsent(boolean)>",
        "<com.ironsource.mediationsdk.BannerManager: void setConsent(boolean)>",
        "<com.ironsource.mediationsdk.BannerSmash: void setConsent(boolean)>",
        "<com.ironsource.adapters.vungle.VungleAdapter: void setConsent(boolean)>",
        "<com.vungle.warren.Vungle: void updateConsentStatus(com.vungle.warren.Vungle$Consent,java.lang.String)>"
    ],
    "ApkPath": "/N/project/android_lib_proj/privacy_impl/apks/com.TecnoveApp.Divulgacion_1.6.apk",
    "CalleeSdkName": "Vungle",
    "CurrentMethod": "<com.ironsource.adapters.vungle.VungleAdapter: void setConsent(boolean)>"
}"""

json_string2 = """{
    "CallerSdkName": "Ironsource",
    "ConditionBlock": [
        "if $r3 != $r2 goto $r5 = staticinvoke <java.lang.Boolean: java.lang.Boolean valueOf(boolean)>($z0)"
    ],
    "PackageName": "com.TecnoveApp.Divulgacion",
    "ApiType": "GDPR",
    "FlawType": "ConditionalEnforcementForWrapper",
    "DelegationPath": [
        "<com.ironsource.mediationsdk.IronSource: void setConsent(boolean)>",
        "<com.ironsource.mediationsdk.IronSourceObject: void setConsent(boolean)>",
        "<com.ironsource.mediationsdk.BannerManager: void setConsent(boolean)>",
        "<com.ironsource.mediationsdk.BannerSmash: void setConsent(boolean)>",
        "<com.ironsource.adapters.vungle.VungleAdapter: void setConsent(boolean)>",
        "<com.vungle.warren.Vungle: void updateConsentStatus(com.vungle.warren.Vungle$Consent,java.lang.String)>"
    ],
    "ApkPath": "/N/project/android_lib_proj/privacy_impl/apks/com.TecnoveApp.Divulgacion_1.6.apk",
    "CalleeSdkName": "Vungle",
    "CurrentMethod": "<com.ironsource.adapters.vungle.VungleAdapter: void setConsent(boolean)>"
}"""

json_string3 = """{
    "CallerSdkName": "Ironsource",
    "ConditionBlock": [
        "if $r3 != $r2 goto $r5 = staticinvoke <java.lang.Boolean: java.lang.Boolean valueOf(boolean)>($z0)", 
    ],
    "PackageName": "com.TecnoveApp.Divulgacion",
    "ApiType": "GDPR",
    "FlawType": "ConditionalEnforcementForWrapper",
    "DelegationPath": [
        "<com.ironsource.mediationsdk.IronSource: void setConsent(boolean)>",
        "<com.ironsource.mediationsdk.IronSourceObject: void setConsent(boolean)>",
        "<com.ironsource.mediationsdk.BannerManager: void setConsent(boolean)>",
        "<com.ironsource.mediationsdk.BannerSmash: void setConsent(boolean)>",
        "<com.ironsource.adapters.vungle.VungleAdapter: void setConsent(boolean)>",
        "<com.vungle.warren.Vungle: void updateConsentStatus(com.vungle.warren.Vungle$Consent,java.lang.String)>"
    ],
    "ApkPath": "/N/project/android_lib_proj/privacy_impl/apks/com.TecnoveApp.Divulgacion_1.6.apk",
    "CalleeSdkName": "Vungle",
    "CurrentMethod": "<com.ironsource.adapters.vungle.VungleAdapter: void setConsent(boolean)>"
}"""

print(load_json_with_check(json_string1))
print(load_json_with_check(json_string2))
try:
    print(load_json_with_check(json_string3))
except:
    raise Exception('can not load: {}'.format(json_string3))