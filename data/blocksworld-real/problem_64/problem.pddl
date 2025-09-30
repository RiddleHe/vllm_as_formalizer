(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        orange - block
        red - block
        green - block
        blue - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (clear purple)
        (ontable orange)
        (on red orange)
        (on green red)
        (clear green)
        (ontable blue)
        (clear blue)
        (ontable yellow)
        (clear yellow)
        ; purple
        ; orange -> red -> green
        ; blue
        ; yellow
    )
    (:goal (and (on red purple)
        (on blue red)
        (on orange blue)
        (on yellow green)))
)
    