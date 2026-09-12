"""
create_sample_policy.py — Generate a sample insurance policy PDF for testing
the exclusion RAG pipeline.

Creates a realistic-looking policy document with:
- Coverage overview and procedures
- Numbered exclusion clauses (including pregnancy, pre-existing conditions, etc.)
- Glossary/definitions section
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import os


def create_sample_policy_pdf(output_path: str = "sample_policy.pdf"):
    """Generate a multi-page sample insurance policy PDF."""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "PolicyTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=20,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading1"],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textTransform="uppercase",
    )
    subheading_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=15,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "PolicyBody",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        leading=14,
    )
    exclusion_style = ParagraphStyle(
        "ExclusionItem",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=10,
        leftIndent=1 * cm,
        alignment=TA_JUSTIFY,
        leading=14,
    )
    definition_style = ParagraphStyle(
        "DefinitionItem",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=10,
        leftIndent=1 * cm,
        alignment=TA_JUSTIFY,
        leading=14,
    )

    story = []

    # ---- Title Page ----
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("COMPREHENSIVE HEALTH INSURANCE POLICY", title_style))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("Policy Document — Terms and Conditions", styles["Heading2"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Effective Date: January 1, 2025", body_style))
    story.append(Paragraph("Policy Series: POL-40100 / POL-10234 / POL-20456 / POL-50321", body_style))
    story.append(Paragraph("Underwritten by: Sample Insurance Corporation", body_style))
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(
        "This document sets out the terms, conditions, exclusions, and definitions "
        "applicable to your health, auto, and home insurance coverage. Please read "
        "this document carefully and retain it for your records.",
        body_style,
    ))
    story.append(PageBreak())

    # ---- Section 1: Coverage Overview ----
    story.append(Paragraph("COVERAGE OVERVIEW", heading_style))
    story.append(Paragraph(
        "This policy provides coverage for the following benefit types, subject to "
        "the terms, conditions, and exclusions set out in this document:",
        body_style,
    ))

    benefits = [
        ("Hospital & Surgical Benefit", "Covers inpatient hospitalisation, surgical procedures, "
         "operating theatre charges, anaesthesia, and post-operative care. Subject to policy "
         "limits and pre-authorisation requirements for elective procedures."),
        ("Supplementary Major Medical (SMM)", "Extended coverage for major medical expenses "
         "exceeding the base Hospital & Surgical limits, including specialist consultations, "
         "advanced diagnostic imaging, and intensive care."),
        ("Clinical Benefit", "Covers outpatient clinical consultations, prescribed medications, "
         "diagnostic tests, and preventive health screenings as specified in the benefit schedule."),
        ("Maternity Benefit", "Coverage for pregnancy-related expenses including prenatal care, "
         "delivery (normal and caesarean), and postnatal care. Subject to a 10-month waiting "
         "period from the coverage start date."),
        ("Dental Benefit", "Covers routine dental examinations, cleanings, fillings, extractions, "
         "and basic restorative dental work as specified in the benefit schedule."),
        ("Network Dental Benefit", "Enhanced dental coverage available exclusively through "
         "participating network dental providers, with reduced co-payments and extended "
         "coverage for orthodontic and periodontic treatments."),
    ]

    for name, desc in benefits:
        story.append(Paragraph(f"<b>{name}</b>", body_style))
        story.append(Paragraph(desc, body_style))
        story.append(Spacer(1, 0.3 * cm))

    story.append(PageBreak())

    # ---- Section 2: Claims Procedures ----
    story.append(Paragraph("CLAIMS PROCEDURES", heading_style))

    story.append(Paragraph("Submission Requirements", subheading_style))
    story.append(Paragraph(
        "All claims must be submitted within ninety (90) days of the date of treatment "
        "or discharge. Claims submitted after this deadline may be declined at the "
        "discretion of the insurer. The following documents are required for claim submission:",
        body_style,
    ))

    docs_required = [
        "Completed claim form signed by the policyholder",
        "Original receipts and invoices from the medical provider",
        "Referral letter from the attending physician (if applicable)",
        "Pre-authorisation confirmation (for elective and scheduled procedures)",
        "Discharge summary (for inpatient hospitalisation claims)",
        "Police report (for accident-related claims)",
        "Copy of the BHN card (if using BHN network providers)",
    ]
    for d in docs_required:
        story.append(Paragraph(f"• {d}", body_style))

    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Pre-Authorisation", subheading_style))
    story.append(Paragraph(
        "Pre-authorisation is required for all elective surgical procedures, inpatient "
        "admissions exceeding 24 hours, and any treatment with an estimated cost exceeding "
        "HKD 50,000. Failure to obtain pre-authorisation may result in a reduction of "
        "benefits payable by up to 50%.",
        body_style,
    ))

    story.append(Paragraph("Claim Processing", subheading_style))
    story.append(Paragraph(
        "Upon receipt of a complete claim submission, the insurer will process the claim "
        "within fifteen (15) business days. Claims requiring additional documentation or "
        "investigation may take up to forty-five (45) business days. The insurer reserves "
        "the right to request additional information or conduct independent medical reviews.",
        body_style,
    ))

    story.append(PageBreak())

    # ---- Section 3: Auto Insurance Procedures ----
    story.append(Paragraph("AUTO INSURANCE COVERAGE", heading_style))
    story.append(Paragraph(
        "Auto insurance coverage provides protection against financial losses arising from "
        "vehicle accidents, theft, and third-party liability. Coverage includes repair costs, "
        "replacement value, towing charges, and rental vehicle expenses during the repair period.",
        body_style,
    ))

    story.append(Paragraph("Accident Reporting", subheading_style))
    story.append(Paragraph(
        "All vehicle accidents must be reported to the insurer within seventy-two (72) hours "
        "of the incident. A police report must be filed for all accidents involving injury, "
        "third-party property damage exceeding HKD 5,000, or suspected criminal activity.",
        body_style,
    ))

    story.append(PageBreak())

    # ---- Section 4: EXCLUSIONS ----
    story.append(Paragraph("EXCLUSIONS", heading_style))
    story.append(Paragraph(
        "The following conditions, treatments, and circumstances are excluded from coverage "
        "under this policy. No benefits shall be payable in respect of any claim arising "
        "directly or indirectly from the following:",
        body_style,
    ))
    story.append(Spacer(1, 0.3 * cm))

    exclusions = [
        "Pre-existing conditions diagnosed or treated within twelve (12) months prior to "
        "the coverage start date, unless specifically declared and accepted by the insurer "
        "at the time of policy inception or renewal.",

        "Cosmetic or aesthetic procedures, including but not limited to facelifts, liposuction, "
        "rhinoplasty, breast augmentation or reduction, hair transplantation, teeth whitening, "
        "and any surgery performed primarily to improve appearance rather than to restore "
        "function or treat a medical condition.",

        "Experimental or investigational treatments, drugs, or medical devices that have not "
        "received regulatory approval from the relevant health authority in the jurisdiction "
        "where the treatment is administered.",

        "Self-inflicted injuries, whether intentional or as a result of attempted suicide, "
        "or injuries sustained while under the influence of alcohol or non-prescribed "
        "controlled substances.",

        "Injuries or illnesses arising from participation in professional sports, extreme sports "
        "(including but not limited to skydiving, bungee jumping, motor racing, mountaineering "
        "above 4,000 metres), or any hazardous recreational activity unless specifically "
        "covered by an additional rider.",

        "Treatment obtained outside the policy territory without prior written approval from "
        "the insurer, except in cases of genuine medical emergency where treatment could not "
        "reasonably be delayed until return to the policy territory.",

        "Dental treatment not specifically covered under the Dental Benefit or Network Dental "
        "Benefit, including but not limited to dental implants, veneers, and full mouth "
        "reconstruction, unless required as a direct result of an accidental injury.",

        "War, civil unrest, terrorism, nuclear contamination, or any military action, whether "
        "declared or undeclared. This exclusion applies regardless of whether the insured "
        "person was a participant or bystander.",

        "Treatment for alcohol or substance abuse, drug addiction, or rehabilitation programs "
        "related to chemical dependency, unless covered under a specific mental health and "
        "substance abuse rider.",

        "Pregnancy, childbirth, miscarriage, abortion, prenatal and postnatal care, and any "
        "complications arising from pregnancy, unless the insured is covered under the "
        "Maternity Benefit and the treatment occurs after the applicable waiting period of "
        "ten (10) months from the coverage start date.",

        "Routine health check-ups, vaccinations, and preventive screenings not specifically "
        "listed in the Clinical Benefit schedule, including annual physical examinations "
        "and wellness programmes.",

        "Congenital conditions, birth defects, and hereditary diseases, unless manifesting "
        "for the first time after the coverage start date and not previously diagnosed or "
        "treated.",

        "Work-related injuries or occupational diseases covered under any workers' compensation "
        "scheme, employer liability insurance, or government-mandated occupational health "
        "programme.",

        "Treatment provided by a family member or any person who ordinarily resides with "
        "the insured, regardless of their medical qualifications.",

        "Any claim where the insured has failed to follow the prescribed treatment plan "
        "of the attending physician, or has discharged themselves from hospital against "
        "medical advice, to the extent that such failure has contributed to the claimed "
        "condition or its worsening.",

        "Hearing aids, spectacles, contact lenses, and other corrective visual or auditory "
        "devices, unless required as a direct result of an accidental injury occurring "
        "during the policy period.",

        "Organ transplantation procedures and related expenses, including donor screening, "
        "organ procurement, and anti-rejection medication, unless specifically covered "
        "under the Supplementary Major Medical benefit with prior written approval.",

        "Alternative medicine treatments including acupuncture, homeopathy, naturopathy, "
        "traditional Chinese medicine, chiropractic treatment, and osteopathy, unless "
        "administered by a licensed medical practitioner and pre-approved by the insurer.",

        "Infertility treatments, assisted reproductive technologies including in-vitro "
        "fertilisation (IVF), intrauterine insemination (IUI), surrogacy arrangements, "
        "and any related hormonal therapy or diagnostic testing.",

        "Travel-related claims including trip cancellation, luggage loss, travel delays, "
        "and medical evacuation, unless covered under a separate travel insurance rider "
        "attached to this policy.",
    ]

    for i, text in enumerate(exclusions, 1):
        story.append(Paragraph(f"{i}. {text}", exclusion_style))

    story.append(PageBreak())

    # ---- Section 5: GLOSSARY / DEFINITIONS ----
    story.append(Paragraph("GLOSSARY AND DEFINITIONS", heading_style))
    story.append(Paragraph(
        "The following terms shall have the meanings set out below when used in this "
        "policy document:",
        body_style,
    ))
    story.append(Spacer(1, 0.3 * cm))

    definitions = [
        ('"Accident"', "means a sudden, unforeseen, and involuntary event caused by external "
         "means resulting in bodily injury, and which occurs independently of any illness "
         "or pre-existing condition of the insured."),

        ('"Benefit Schedule"', "means the table of benefits, limits, deductibles, and "
         "co-payment percentages applicable to each category of coverage as set out in "
         "the policy schedule issued to the policyholder."),

        ('"BHN Card"', "means the Benefit Healthcare Network identification card issued "
         "to the insured for use at participating network healthcare providers, enabling "
         "direct billing and cashless claim settlement."),

        ('"Congenital Condition"', "means any medical condition, disease, or abnormality "
         "that is present at birth, whether diagnosed at the time of birth or subsequently, "
         "including but not limited to genetic disorders and birth defects."),

        ('"Coverage Start Date"', "means the date on which the insured's coverage under "
         "this policy commences, as specified in the policy schedule."),

        ('"Deductible"', "means the fixed amount or percentage of the claim that the "
         "insured must pay out of pocket before the insurer's liability begins."),

        ('"Elective Procedure"', "means any medical or surgical procedure that is scheduled "
         "in advance and is not required as an immediate response to a medical emergency."),

        ('"Hospital"', "means a licensed medical facility that provides inpatient care, "
         "has organised medical staff, provides 24-hour nursing services, and has facilities "
         "for diagnosis, treatment, and surgery."),

        ('"Maternity Benefit"', "means the specific coverage provision for pregnancy-related "
         "expenses, subject to the waiting period and conditions specified in this policy."),

        ('"Policy Territory"', "means the geographical area within which the insured is "
         "covered, as specified in the policy schedule. Unless otherwise stated, the policy "
         "territory is limited to Hong Kong SAR."),

        ('"Pre-existing Condition"', "means any illness, injury, disease, or physical "
         "condition for which symptoms existed, medical advice or treatment was received, "
         "or medication was prescribed within the twelve (12) months immediately preceding "
         "the coverage start date."),

        ('"Waiting Period"', "means the period of time from the coverage start date during "
         "which certain benefits are not payable, as specified in the benefit schedule for "
         "each applicable benefit category."),
    ]

    for term, defn in definitions:
        story.append(Paragraph(f"{term} {defn}", definition_style))

    # ---- Build the PDF ----
    doc.build(story)
    print(f"Sample policy PDF created: {os.path.abspath(output_path)}")
    return output_path


if __name__ == "__main__":
    create_sample_policy_pdf()
