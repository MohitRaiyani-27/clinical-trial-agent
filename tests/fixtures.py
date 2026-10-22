"""SYNTHETIC study records in the ClinicalTrials.gov API v2 shape.

These are invented for offline unit tests only (NCT IDs in the NCT0000000x range,
'Example' sponsors) and do not describe real studies.
"""

STUDY_A = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Example Drug X in Plaque Psoriasis"},
        "statusModule": {"overallStatus": "RECRUITING", "startDateStruct": {"date": "2025-03"},
                         "primaryCompletionDateStruct": {"date": "2027-06"}},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Example Pharma Inc"}},
        "descriptionModule": {"briefSummary": "A study of Example Drug X versus placebo in adults with moderate to severe plaque psoriasis."},
        "conditionsModule": {"conditions": ["Plaque Psoriasis"]},
        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"],
                         "enrollmentInfo": {"count": 420, "type": "ESTIMATED"}},
        "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Example Drug X"},
                                                      {"type": "DRUG", "name": "Placebo"}]},
        "outcomesModule": {
            "primaryOutcomes": [{"measure": "Proportion of participants achieving PASI 75",
                                 "timeFrame": "Week 16"}],
            "secondaryOutcomes": [{"measure": "Change in DLQI score", "timeFrame": "Week 16"}],
        },
        "eligibilityModule": {
            "eligibilityCriteria": (
                "Inclusion Criteria:\n\n* Adults aged 18 years or older\n"
                "* Diagnosis of plaque psoriasis for at least 6 months\n"
                "* PASI score of 12 or greater\n\n"
                "Exclusion Criteria:\n\n* Prior treatment with any biologic therapy targeting IL-17\n"
                "* Active tuberculosis infection\n* Pregnant or breastfeeding women"
            ),
            "sex": "ALL", "minimumAge": "18 Years", "maximumAge": "75 Years",
        },
    }
}

STUDY_B = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT00000002", "briefTitle": "Example Drug Y Dose Finding in Psoriasis"},
        "statusModule": {"overallStatus": "COMPLETED"},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Example Biotech LLC"}},
        "conditionsModule": {"conditions": ["Psoriasis"]},
        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE2"],
                         "enrollmentInfo": {"count": 120, "type": "ACTUAL"}},
        "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Example Drug Y"}]},
        "outcomesModule": {"primaryOutcomes": [{"measure": "Percent change in PASI from baseline",
                                                "timeFrame": "Week 12"}]},
        "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria:\n\n* Age 18 to 65\n",
                              "minimumAge": "18 Years", "maximumAge": "65 Years"},
    }
}

STUDIES = [STUDY_A, STUDY_B]
