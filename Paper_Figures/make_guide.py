from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parent
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyText2',fontName='Helvetica',fontSize=10.5,leading=15,spaceAfter=9,textColor=colors.HexColor('#243447')))
styles['Title'].textColor=colors.HexColor('#2166ac');styles['Title'].fontSize=24;styles['Title'].leading=29
styles['Heading1'].textColor=colors.HexColor('#2166ac');styles['Heading1'].fontSize=17
styles['Heading2'].fontSize=12;styles['Heading2'].spaceBefore=12
story=[]
def p(t):story.append(Paragraph(t,styles['BodyText2']))
def h(t):story.append(Paragraph(t,styles['Heading1']));story.append(Spacer(1,10))
def table(rows,widths):
    data=[[Paragraph(escape(str(c)),styles['BodyText2']) for c in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f1f8')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d9e1e8'))]));story.append(t)
def image(name):
    path=ROOT/'architecture'/f'{name}.png';w,hh=ImageReader(str(path)).getSize();story.append(Image(str(path),width=490,height=490*hh/w));story.append(Spacer(1,10))

story.append(Paragraph('FedContextTrust-V2X',styles['Title']));story.append(Spacer(1,8))
h('Paper figures and interpretation guide')
p('Six methodology diagrams and a reproducible plotting package for your completed federated experiment. Use the diagrams now; generate the measured result figures from your own saved prediction and training files.')
p('<b>Status:</b> the supplied README describes the implementation. Actual prediction CSVs, training logs and evaluation metrics were not supplied for this package, so no measured results or accuracy claims have been filled in.')
table([['Included','Purpose'],['6 architecture diagrams','Data preparation, one RSU, federation, OBU inference, calibration and causal windows.'],['Up to 23 result figures','Training losses, detection metrics, errors, stream comparisons, attack groups and communication costs.'],['Automatic interpretation report','The plotting script writes measured values, captions and interpretation beside each available graph.'],['PDF / SVG / 300 dpi PNG','Vector exports for papers; raster exports for Word and slides.']],[150,340])
story.append(Spacer(1,15));p('<b>Recommended paper set:</b> architecture, local training losses, ROC and precision-recall, both confusion matrices, stream-subset comparison, attack-group recall, and communication cost. Extra diagnostics can go in supplementary material.')
p('Keep claims proportional to the evidence: low training loss is not proof of high detection accuracy, and laptop measurements are not physical OBU deployment results.')
story.append(PageBreak())
h('Run it on your saved experiment')
p('1. Copy the Paper_Figures folder beside src, data, runs and pyproject.toml in the full project. Open PowerShell there.')
p('2. Install the dependencies using the command in README.md. Run generate_figures.py with the evaluation directory, matching calibration.json, all RSU client_rounds.csv files and cloud communication.csv.')
p('3. Open the generated report.html. It shows every available result graph with its caption and interpretation. Missing inputs are reported; no missing result is synthesized.')
p('4. Insert PDF or SVG figures into the paper where supported. For Word, use the 300 dpi PNG files and keep axis text readable at final printed size.')
table([['Argument','What it selects'],['--evaluation','Held-out predictions.csv.gz and metrics.csv from finalize.'],['--calibration','The saved thresholds from the same finalized model.'],['--clients','A quoted wildcard matching only RSU logs from this experiment.'],['--cloud','The cloud communication.csv with per-round duration.'],['--out','A new output directory; existing results are preserved.']],[130,360])
story.append(Spacer(1,15));p('The script validates scores and labels, checks that supplied thresholds reproduce the saved decisions, and rejects duplicate prediction records. Its figure manifest records input-file hashes. Keep all supplied client logs from one federated run.')
p('No new training is needed. The script plots existing predictions. It does not change model weights, thresholds or saved detector decisions.')
story.append(PageBreak())

diags=[
('D1_data_and_experiment_flow','Data preparation and experimental separation',
 'Multiple extracted JSON files enter validation and causal window construction. Official test files remain separate. Benign training windows are partitioned into RSU shards. Distinct validation partitions supply the error reference and score calibration.',
 'This separation is what allows final test results to measure performance on data not used to fit weights or thresholds. The preparation machine can access the benchmark; the weights-only claim applies to the network training phase.'),
('D2_one_RSU_models','What happens inside one RSU',
 'Three independent neural models share an optimization step. The movement GRU predicts a six-feature next movement vector. The timing GRU autoencoder reconstructs a ten-by-seven sequence. The context MLP autoencoder reconstructs a ten-slot vector.',
 'Each model provides MSE in scaled feature space. Adam minimizes their summed mean errors on benign windows. The three road-related context slots are zero in the network implementation; active context comes from receiver and neighbour observations.'),
('D3_federated_learning','How the cloud combines local learning',
 'All registered RSUs download the same global model, train their local copies and upload updated weights. The cloud waits for every client and averages matching tensors with weights proportional to local training-window counts n_k.',
 'This is synchronous full-participation FedAvg. Clients fetch the updated global weights for the next round. The prototype uses separate processes and actual HTTP exchange. It does not implement secure aggregation, differential privacy or poisoning defence.'),
('D4_OBU_inference','How a message gets its decision',
 'At the receiver, the new message updates temporal and neighbour histories. The three models return errors; benign-reference ranks put them on a comparable scale. Their mean is the fused anomaly score.',
 'Two fixed calibrated thresholds assign Trusted, Uncertain or Suspicious. Insufficient history also produces Uncertain without a model score. Inference does not retrain models. OBU placement describes the intended deployment; current execution is a PC simulation.'),
('D5_offline_calibration','How normal data sets the thresholds',
 'Frozen final weights score a benign reference partition and a different benign calibration partition. Each error becomes its empirical reference rank. Calibration scores yield the lower and upper quantile thresholds.',
 'The default quantiles are 0.95 and 0.99. They are not literal thresholds of 0.95 and 0.99 on every saved detector, and they do not guarantee exact future false-positive rates. Calibration is offline, not federated; deployment of the finalized package is a separate step.'),
('D6_window_and_target','Why the three model inputs differ',
 'For a next-message example, movement uses the previous ten sender observations to predict observation eleven. Timing uses the latest ten timing vectors including observation eleven. Context describes observation eleven using already available neighbours.',
 'Movement is next-step prediction. The other two tasks are reconstruction. All use past and currently received observations, never future messages. Window lengths in this diagram show the default configuration.'),
]
for name,title,caption,meaning in diags:
    h(title);image(name);p('<b>Caption:</b> '+caption);p('<b>Interpretation:</b> '+meaning);story.append(PageBreak())

h('Reading detection performance')
table([['Figure','Theoretical interpretation'],['ROC curve','True-positive rate against false-positive rate as the score threshold changes. A curve toward the upper-left gives better ranking; report performance at saved thresholds too.'],['Precision-recall curve','Precision tells how reliable alerts are; recall tells how many attacks are found. Average precision summarizes this curve. Its baseline depends on attack prevalence.'],['Strict / alert confusion matrices','Strict treats only Suspicious as positive. Alert treats Uncertain and Suspicious as positive. Both retain every scored window in evaluation.'],['Threshold sensitivity','Shows the detection/false-alarm tradeoff. This is descriptive test analysis, not a valid way to tune thresholds after seeing test labels.'],['Score histogram and ECDF','Separate class distributions explain threshold behaviour. Overlap produces missed attacks or false alarms. Scores are ranks, not attack probabilities.'],['Three-decision proportions','Makes uncertainty visible within benign and attack windows. There is no third ground-truth class called Uncertain.']],[145,345])
story.append(Spacer(1,10));p('<b>Metric definitions:</b> Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN); FPR = FP/(FP+TN); accuracy = (TP+TN)/N. Undefined denominators must stay undefined.')
story.append(PageBreak())
h('Reading model and federation diagnostics')
table([['Figure','What the evidence supports'],['Per-RSU training losses','Whether local benign-data fitting improves. Small increases after global weight averaging are possible. These losses are not global validation losses.'],['Weighted local loss','A sample-weighted summary of client training losses, not a fresh evaluation of the averaged checkpoint.'],['Stream MSE / rank distributions','Whether each type of evidence separates benign and attack windows. Raw MSE scales differ; ranks are more comparable.'],['Stream-subset comparison','Whether using additional saved streams helps at calibrated thresholds. This is post-training subset analysis, not separately retrained ablation.'],['Attack-group recall','Which attack groups are detected or missed. Attack-only groups cannot establish precision or benign false-positive rate.'],['Sender score timeline','A case study of scores crossing thresholds. The selected example is not representative evidence on its own.'],['Payload / duration curves','Recorded training communication and runtime costs. Payload excludes protocol overhead; training duration is not inference latency.']],[145,345])
story.append(PageBreak())
h('What to write, and what still needs evidence')
p('Use a result sentence only after inserting measured values from your run:')
p('"On [number] held-out test windows, the strict policy achieved precision [ ], recall [ ], F1 [ ] and FPR [ ]. The alert policy changed recall to [ ] and FPR to [ ]. The full fusion system obtained ROC-AUC [ ] and average precision [ ]."')
p('"The local training curves show [observed trend]. Their fluctuations reflect local optimization across federated rounds and do not directly measure test classification performance. The recorded model-body traffic over [ ] rounds was [ ] MiB, excluding protocol overhead."')
p('Leave these entries empty until you run the plotting script. If performance is poor for a particular attack, report that limitation and investigate its evidence streams instead of hiding the group.')
p('<b>Extra experiments:</b> repeated seeds; matched local-only and centralized baselines; session-aware uncertainty estimates; retrained stream ablations; held-out attack families; actual OBU profiling. Never claim improvements over the five related papers from unrelated published numbers.')
p('<b>Missing curves:</b> accuracy versus round is unavailable from existing network logs. A new experiment must evaluate saved global checkpoints on a fixed validation set each round. Do not compute accuracy as 1 minus training MSE. Final test data should remain reserved for the locked detector.')
p('<b>Dependence:</b> overlapping windows from a session are correlated. Confidence intervals should account for sessions/runs rather than treating each window as an independent trial. The current package reports descriptive results without invented confidence intervals.')
p('<b>References:</b><br/>Implementation: supplied README_FEDERATED.md.<br/>FedAvg: McMahan et al., AISTATS 2017, proceedings.mlr.press/v54/mcmahan17a.html.<br/>Metrics: scikit-learn.org/stable/modules/model_evaluation.html.')

def footer(c,d):
    c.saveState();c.setStrokeColor(colors.HexColor('#d9e1e8'));c.line(48,42,547,42);c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#64748b'));c.drawString(48,28,'FedContextTrust-V2X | Figure guide | Measured results supplied separately');c.drawRightString(547,28,str(d.page));c.restoreState()
SimpleDocTemplate(str(ROOT/'Paper_Figures_Guide.pdf'),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=55).build(story,onFirstPage=footer,onLaterPages=footer)
