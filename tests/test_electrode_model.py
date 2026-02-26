# tests/test_electrode_model.py
import sys
import os

# Add the module source to the path so we can import without Slicer
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "SEEGFellow", "SEEGFellow")
)

from SEEGFellowLib.electrode_model import Contact, Electrode, ElectrodeParams


class TestContact:
    def test_create_contact(self):
        c = Contact(index=1, position_ras=(10.0, 20.0, 30.0), label="A1")
        assert c.index == 1
        assert c.position_ras == (10.0, 20.0, 30.0)
        assert c.label == "A1"

    def test_default_label_is_empty(self):
        c = Contact(index=3, position_ras=(0.0, 0.0, 0.0))
        assert c.label == ""


class TestElectrodeParams:
    def test_params_without_gaps(self):
        p = ElectrodeParams(
            contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8
        )
        assert p.gap_spacing is None
        assert p.contacts_per_group is None

    def test_params_with_gaps(self):
        p = ElectrodeParams(
            contact_length=2.0,
            contact_spacing=3.5,
            contact_diameter=0.8,
            gap_spacing=10.0,
            contacts_per_group=6,
        )
        assert p.gap_spacing == 10.0
        assert p.contacts_per_group == 6


class TestElectrode:
    def test_create_electrode(self):
        params = ElectrodeParams(
            contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8
        )
        contacts = [
            Contact(index=i, position_ras=(float(i), 0.0, 0.0)) for i in range(1, 7)
        ]
        e = Electrode(
            name="A",
            params=params,
            contacts=contacts,
            trajectory_direction=(1.0, 0.0, 0.0),
        )
        assert e.name == "A"
        assert e.num_contacts == 6
        assert len(e.contacts) == 6

    def test_assign_labels(self):
        params = ElectrodeParams(
            contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8
        )
        contacts = [
            Contact(index=i, position_ras=(float(i), 0.0, 0.0)) for i in range(1, 4)
        ]
        e = Electrode(
            name="",
            params=params,
            contacts=contacts,
            trajectory_direction=(1.0, 0.0, 0.0),
        )
        e.assign_labels("LT")
        assert e.name == "LT"
        assert e.contacts[0].label == "LT1"
        assert e.contacts[2].label == "LT3"
