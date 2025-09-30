(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        purple - block
        blue - block
        orange - block
        red - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (on purple yellow)
        (on blue purple)
        (clear blue)
        (ontable orange)
        (on red orange)
        (on green red)
        (clear green)
        ; yellow -> purple -> blue
        ; orange -> red -> green
    )
    (:goal (and (on blue yellow)
        (on purple green)
        (on red purple)
        (on orange red)))
)
    