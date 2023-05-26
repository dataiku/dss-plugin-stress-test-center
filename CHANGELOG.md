# Changelog

## Version 1.1.7 - 2023-05-26
- Remove code env

## Version 1.0.7 - 2023-04-19
- Webapp: Fix issue when at least one class is not present in the target mapping of the modeling task

## Hotfix 1.0.6 - 2022-08-10
- Webapp: Fix typo

## Version 1.0.5 - 2022-07-27
- Webapp: Only display model view for tasks using a Python backend
- Webapp: Make the model view py2 compatible
- Webapp: Use a better ModalService
- Webapp: Use a simpler way to load the models from DSS

## Version 1.0.4 - 2022-06-09
- Webapp: Only display model view for binary, multiclass & regression tasks
- Webapp: Use contextual code env

## Version 1.0.3 - 2022-01-27
- New test in webapp: feature distribution shift
- Support of sample weights in metric computations
- Webapp: Fix the case where the target has missing values
- UI fixes on tooltips & dropdowns

## Version 1.0.2 - 2022-01-12
- UI fixes (empty section in result page, enabling submission of settings form when pressing enter)

## Version 1.0.1 - 2022-01-10
- UI enhancements (Safari compatibility, consistent wording in settings, tooltip to clarify why the Run tests button is disabled, warning in results for "not relevant" stress tests)
- Ability for users to pick the evaluation metric they want to be used for the performance variation computation
- Cleaning of the code env requirements

## Version 1.0.0 - Initial release - 2021-12-24
- Initial release
- Model view component featuring three stress tests
- Recipe component featuring two stress tests
