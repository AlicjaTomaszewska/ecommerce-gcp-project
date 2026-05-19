const projectConfig = dataform.projectConfig;

const snapshotDate = projectConfig.vars.snapshot_date;
const bronzeDataset = projectConfig.vars.bronze_dataset;
const silverDataset = projectConfig.vars.silver_dataset;
const goldDataset = projectConfig.vars.gold_dataset;

module.exports = {
  snapshotDate,
  bronzeDataset,
  silverDataset,
  goldDataset
};
