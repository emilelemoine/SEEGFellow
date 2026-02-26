# SEEGFellow/SEEGFellow/SEEGFellowLib/electrode_model.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Contact:
    """A single electrode contact.

    Example::

        c = Contact(index=1, position_ras=(10.0, 20.0, 30.0), label="A1")
    """

    index: int  # 1-based (1 = deepest/mesial)
    position_ras: tuple[float, float, float]  # (R, A, S) in mm
    label: str = ""


@dataclass
class ElectrodeParams:
    """Physical parameters of an electrode model.

    Example::

        # Standard electrode without gaps
        p = ElectrodeParams(contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8)

        # Electrode with gap between groups of 6 contacts
        p = ElectrodeParams(
            contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8,
            gap_spacing=10.0, contacts_per_group=6,
        )
    """

    contact_length: float  # mm
    contact_spacing: float  # mm center-to-center
    contact_diameter: float  # mm
    gap_spacing: float | None = None  # mm center-to-center between groups
    contacts_per_group: int | None = None  # contacts per group


@dataclass
class Electrode:
    """A detected electrode with its contacts.

    Example::

        e = Electrode(
            name="A",
            params=ElectrodeParams(contact_length=2.0, contact_spacing=3.5, contact_diameter=0.8),
            contacts=[Contact(index=1, position_ras=(10.0, 20.0, 30.0))],
            trajectory_direction=(1.0, 0.0, 0.0),
        )
        e.assign_labels("A")  # contacts become A1, A2, ...
    """

    name: str
    params: ElectrodeParams
    contacts: list[Contact] = field(default_factory=list)
    trajectory_direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    markups_node_id: str = ""

    @property
    def num_contacts(self) -> int:
        return len(self.contacts)

    def assign_labels(self, name: str) -> None:
        """Set electrode name and update all contact labels."""
        self.name = name
        for contact in self.contacts:
            contact.label = f"{name}{contact.index}"
