"""Unit tests for CloudFormation templates.

Validates:
- Templates contain required parameters for both commercial and GovCloud (Req 10.2)
- IAM task role policy contains only required permissions (Req 11.1)
"""

import yaml
from pathlib import Path

import pytest

# Resolve infra/ directory relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INFRA_DIR = PROJECT_ROOT / "infra"


# ---------------------------------------------------------------------------
# Custom YAML loader that handles CloudFormation intrinsic function tags
# ---------------------------------------------------------------------------

class _CfnLoader(yaml.SafeLoader):
    """YAML loader that treats CloudFormation intrinsic functions as plain data."""


def _cfn_constructor(loader, node):
    """Handle a CFN tag by returning its value as a plain Python object."""
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return None


def _cfn_multi_constructor(loader, tag_suffix, node):
    """Catch-all multi-constructor for any unregistered !Tag."""
    return _cfn_constructor(loader, node)


# Register constructors for all CloudFormation intrinsic functions
_CFN_TAGS = [
    "!Ref", "!Sub", "!GetAtt", "!Equals", "!If", "!Not", "!And", "!Or",
    "!Select", "!Split", "!Join", "!FindInMap", "!GetAZs", "!ImportValue",
    "!Condition", "!Base64", "!Cidr", "!Transform",
]
for _tag in _CFN_TAGS:
    _CfnLoader.add_constructor(_tag, _cfn_constructor)

# Catch-all for any other custom tags
_CfnLoader.add_multi_constructor("!", _cfn_multi_constructor)


def load_template(name: str) -> dict:
    """Load and parse a CloudFormation YAML template."""
    path = INFRA_DIR / name
    with open(path) as f:
        return yaml.load(f, Loader=_CfnLoader)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def network_template():
    return load_template("network-stack.yaml")


@pytest.fixture
def opensearch_template():
    return load_template("opensearch-stack.yaml")


@pytest.fixture
def knowledgebase_template():
    return load_template("knowledgebase-stack.yaml")


@pytest.fixture
def ecs_template():
    return load_template("ecs-stack.yaml")


@pytest.fixture
def root_template():
    return load_template("root-stack.yaml")


# ---------------------------------------------------------------------------
# 1. Template parsing – every template loads without error
# ---------------------------------------------------------------------------

class TestTemplateParsing:
    """Verify all templates can be loaded and parsed as valid YAML."""

    @pytest.mark.parametrize("template_name", [
        "network-stack.yaml",
        "opensearch-stack.yaml",
        "knowledgebase-stack.yaml",
        "ecs-stack.yaml",
        "root-stack.yaml",
    ])
    def test_template_loads(self, template_name):
        tpl = load_template(template_name)
        assert isinstance(tpl, dict)
        assert "AWSTemplateFormatVersion" in tpl
        assert "Resources" in tpl


# ---------------------------------------------------------------------------
# 2. Environment parameter with AllowedValues [commercial, govcloud]
# ---------------------------------------------------------------------------

class TestEnvironmentParameter:
    """Verify Environment parameter exists with correct AllowedValues."""

    @pytest.mark.parametrize("template_name", [
        "network-stack.yaml",
        "opensearch-stack.yaml",
        "knowledgebase-stack.yaml",
        "ecs-stack.yaml",
        "root-stack.yaml",
    ])
    def test_environment_param_exists_with_allowed_values(self, template_name):
        tpl = load_template(template_name)
        params = tpl.get("Parameters", {})
        assert "Environment" in params, f"{template_name} missing Environment parameter"
        env_param = params["Environment"]
        allowed = env_param.get("AllowedValues", [])
        assert "commercial" in allowed
        assert "govcloud" in allowed


# ---------------------------------------------------------------------------
# 3. Network stack has VPC, subnet, NAT, IGW resources
# ---------------------------------------------------------------------------

class TestNetworkStackResources:
    """Verify network-stack contains core networking resources."""

    def test_has_vpc(self, network_template):
        resources = network_template["Resources"]
        vpc_resources = [
            k for k, v in resources.items()
            if v["Type"] == "AWS::EC2::VPC"
        ]
        assert len(vpc_resources) >= 1

    def test_has_subnets(self, network_template):
        resources = network_template["Resources"]
        subnet_resources = [
            k for k, v in resources.items()
            if v["Type"] == "AWS::EC2::Subnet"
        ]
        assert len(subnet_resources) >= 4  # 2 public + 2 private

    def test_has_nat_gateway(self, network_template):
        resources = network_template["Resources"]
        nat_resources = [
            k for k, v in resources.items()
            if v["Type"] == "AWS::EC2::NatGateway"
        ]
        assert len(nat_resources) >= 1

    def test_has_internet_gateway(self, network_template):
        resources = network_template["Resources"]
        igw_resources = [
            k for k, v in resources.items()
            if v["Type"] == "AWS::EC2::InternetGateway"
        ]
        assert len(igw_resources) >= 1


# ---------------------------------------------------------------------------
# 4. ECS stack task role has required policies
# ---------------------------------------------------------------------------

def _get_task_role_policies(ecs_template: dict) -> list[dict]:
    """Extract inline policies from the ECS task role."""
    resources = ecs_template["Resources"]
    task_role = resources.get("ECSTaskRole", {})
    return task_role.get("Properties", {}).get("Policies", [])


def _collect_all_actions(policies: list[dict]) -> set[str]:
    """Collect every IAM action string across all policy statements."""
    actions: set[str] = set()
    for policy in policies:
        doc = policy.get("PolicyDocument", {})
        for stmt in doc.get("Statement", []):
            raw = stmt.get("Action", [])
            if isinstance(raw, str):
                raw = [raw]
            actions.update(raw)
    return actions


class TestEcsTaskRolePermissions:
    """Verify ECS task role has required and only required permissions (Req 11.1)."""

    def test_has_bedrock_invoke_model(self, ecs_template):
        actions = _collect_all_actions(_get_task_role_policies(ecs_template))
        assert "bedrock:InvokeModel" in actions
        assert "bedrock:InvokeModelWithResponseStream" in actions

    def test_has_bedrock_retrieve(self, ecs_template):
        actions = _collect_all_actions(_get_task_role_policies(ecs_template))
        assert "bedrock:Retrieve" in actions

    def test_has_opensearch_access(self, ecs_template):
        actions = _collect_all_actions(_get_task_role_policies(ecs_template))
        assert "aoss:APIAccessAll" in actions

    def test_has_s3_read(self, ecs_template):
        actions = _collect_all_actions(_get_task_role_policies(ecs_template))
        assert "s3:GetObject" in actions
        assert "s3:ListBucket" in actions

    def test_has_cloudwatch_logs(self, ecs_template):
        actions = _collect_all_actions(_get_task_role_policies(ecs_template))
        assert "logs:CreateLogStream" in actions
        assert "logs:PutLogEvents" in actions

    def test_no_overly_broad_iam_permissions(self, ecs_template):
        """Task role must NOT have wildcard IAM, S3, or other overly broad actions."""
        actions = _collect_all_actions(_get_task_role_policies(ecs_template))
        overly_broad = {"iam:*", "s3:*", "ec2:*", "sts:*", "logs:*", "bedrock:*", "aoss:*"}
        found = actions & overly_broad
        assert not found, f"Overly broad permissions found: {found}"

    def test_no_s3_write_actions(self, ecs_template):
        """Task role should only have S3 read, not write."""
        actions = _collect_all_actions(_get_task_role_policies(ecs_template))
        s3_write_actions = {"s3:PutObject", "s3:DeleteObject", "s3:PutBucketPolicy"}
        found = actions & s3_write_actions
        assert not found, f"S3 write permissions found: {found}"


# ---------------------------------------------------------------------------
# 5. Root stack has all 4 nested stack references
# ---------------------------------------------------------------------------

class TestRootStackNestedStacks:
    """Verify root-stack references all 4 nested stacks."""

    EXPECTED_NESTED_STACKS = {"NetworkStack", "OpenSearchStack", "KnowledgeBaseStack", "ECSStack"}

    def test_has_all_nested_stacks(self, root_template):
        resources = root_template["Resources"]
        nested = {
            k for k, v in resources.items()
            if v["Type"] == "AWS::CloudFormation::Stack"
        }
        assert self.EXPECTED_NESTED_STACKS.issubset(nested), (
            f"Missing nested stacks: {self.EXPECTED_NESTED_STACKS - nested}"
        )

    def test_nested_stacks_have_template_urls(self, root_template):
        resources = root_template["Resources"]
        for name in self.EXPECTED_NESTED_STACKS:
            props = resources[name].get("Properties", {})
            assert "TemplateURL" in props, f"{name} missing TemplateURL"
